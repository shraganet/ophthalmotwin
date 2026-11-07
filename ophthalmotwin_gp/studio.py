from __future__ import annotations

import os
from collections.abc import Sequence

import configargparse
import torch

from ophthalmotwin_gp.bench import atomic_load, count_parameters, get_logger, pick_device, set_seed
from ophthalmotwin_gp.cartoon import Run, load_run
from ophthalmotwin_gp.easel import Trainer, assemble, gather_predictions
from ophthalmotwin_gp.lightbox.raps import RapsCalibrator
from ophthalmotwin_gp.lightbox.readings import (
    auroc_ovr,
    coverage,
    macro_f1,
    macro_sensitivity,
    mean_set_size,
    top_k_accuracy,
)
from ophthalmotwin_gp.selection.stock import fold_indices

LOGGER = get_logger()


def _build_parser() -> configargparse.ArgumentParser:
    parser = configargparse.ArgumentParser(
        prog="ophthalmotwin-gp",
        description="dual-stream genotype-phenotype retinal twin atelier",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: configargparse.ArgumentParser) -> None:
        sub.add_argument("--config", required=True, help="cartoon yaml path")
        sub.add_argument("--set", action="append", default=[], help="dotted override key=value")
        sub.add_argument(
            "--device", default="auto", env_var="OPHTHALMO_DEVICE", help="auto|cpu|cuda"
        )

    cut = subparsers.add_parser("cut", help="prepare cohort and variant graph")
    add_common(cut)

    glaze = subparsers.add_parser("glaze", help="train the twin")
    add_common(glaze)
    glaze.add_argument("--out-dir", default=None, env_var="OPHTHALMO_OUT_DIR")

    inspect = subparsers.add_parser("inspect", help="evaluate a checkpoint")
    add_common(inspect)
    inspect.add_argument("--ckpt", default=None, env_var="OPHTHALMO_CKPT")

    hold = subparsers.add_parser("hold", help="infer a single twin state")
    add_common(hold)
    hold.add_argument("--ckpt", default=None, env_var="OPHTHALMO_CKPT")

    trace = subparsers.add_parser("trace", help="export the imaging stream to onnx")
    add_common(trace)
    trace.add_argument("--output", default="runs/clearstream.onnx")
    return parser


def _resolve_run(args: configargparse.Namespace) -> Run:
    overrides = list(args.set)
    overrides.append(f"firing.device={args.device}")
    if getattr(args, "out_dir", None):
        overrides.append(f"firing.out_dir={args.out_dir}")
    return load_run(args.config, overrides)


def _load_ckpt(model: torch.nn.Module, run: Run, explicit: str | None) -> None:
    path = explicit or os.path.join(run.firing.out_dir, "twin.pt")
    payload = atomic_load(path)
    model.load_state_dict(payload["model"])


def cmd_cut(args: configargparse.Namespace) -> None:
    run = _resolve_run(args)
    bundle = assemble(run)
    folds = {f: len(fold_indices(bundle.patients, f)) for f in ("train", "cal", "test")}
    LOGGER.info(
        "cohort %d patients | graph %d nodes | folds %s | genes %d",
        len(bundle.patients),
        bundle.graph.num_nodes,
        folds,
        run.cohort.n_genes,
    )


def cmd_glaze(args: configargparse.Namespace) -> None:
    run = _resolve_run(args)
    bundle = assemble(run)
    trainer = Trainer(bundle.model, run)
    history = trainer.fit(bundle.patients, bundle.graph)
    path = trainer.save_checkpoint(len(history))
    LOGGER.info(
        "trained %d steps | final loss %.4f | params %d | checkpoint %s",
        len(history),
        history[-1] if history else float("nan"),
        count_parameters(bundle.model),
        path,
    )


def cmd_inspect(args: configargparse.Namespace) -> None:
    run = _resolve_run(args)
    bundle = assemble(run)
    _load_ckpt(bundle.model, run, args.ckpt)
    device = pick_device(run.firing.device)
    cal = gather_predictions(
        bundle.model, bundle.graph, bundle.patients, "cal", run.firing.batch_size, device
    )
    test = gather_predictions(
        bundle.model, bundle.graph, bundle.patients, "test", run.firing.batch_size, device
    )
    calibrator = RapsCalibrator(
        alpha=run.firing.alpha, lam=run.firing.raps_lambda, kreg=run.firing.raps_kreg
    )
    calibrator.calibrate(cal.gene_probs, cal.gene_labels)
    sets = calibrator.predict(test.gene_probs)
    LOGGER.info("Top-5            %.3f", top_k_accuracy(test.gene_probs, test.gene_labels, 5))
    LOGGER.info("AUROC (ovr)      %.3f", auroc_ovr(test.gene_probs, test.gene_labels))
    LOGGER.info("macro F1         %.3f", macro_f1(test.gene_probs, test.gene_labels))
    LOGGER.info("macro sens       %.3f", macro_sensitivity(test.gene_probs, test.gene_labels))
    LOGGER.info("coverage @0.90   %.3f", coverage(sets, test.gene_labels))
    LOGGER.info("mean set size    %.2f", mean_set_size(sets))


def cmd_hold(args: configargparse.Namespace) -> None:
    run = _resolve_run(args)
    bundle = assemble(run)
    _load_ckpt(bundle.model, run, args.ckpt)
    device = pick_device(run.firing.device)
    from ophthalmotwin_gp.selection.panes import collate_streams

    test_ids = fold_indices(bundle.patients, "test")
    patient = bundle.patients[test_ids[0]]
    batch = collate_streams([patient]).to(device)
    bundle.model.to(device).eval()
    with torch.no_grad():
        output = bundle.model(batch, bundle.graph.to(device))
    probs = torch.softmax(output.gene_logits, dim=1)[0]
    top = torch.topk(probs, k=5)
    LOGGER.info("patient %s | true gene %d", patient.patient_id, patient.gene)
    LOGGER.info("top-5 genes %s", [int(i) for i in top.indices.tolist()])
    LOGGER.info("prognosis quantiles %s", [round(float(x), 3) for x in output.prog_quantiles[0]])
    LOGGER.info("early-onset prob %.3f", float(torch.sigmoid(output.early_logit)[0]))


def cmd_trace(args: configargparse.Namespace) -> None:
    run = _resolve_run(args)
    set_seed(run.cohort.seed)
    bundle = assemble(run)
    stream = bundle.model.longitudinal.eval()
    c = run.cohort
    images = torch.zeros(1, c.max_visits, c.img_channels, c.img_size, c.img_size)
    deltas = torch.zeros(1, c.max_visits)
    mask = torch.ones(1, c.max_visits, dtype=torch.bool)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(
        stream,
        (images, deltas, mask),
        args.output,
        input_names=["images", "deltas", "visit_mask"],
        output_names=["z_img", "visit_weights"],
        dynamo=False,
        opset_version=17,
    )
    LOGGER.info("exported imaging stream to %s", args.output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "cut": cmd_cut,
        "glaze": cmd_glaze,
        "inspect": cmd_inspect,
        "hold": cmd_hold,
        "trace": cmd_trace,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
