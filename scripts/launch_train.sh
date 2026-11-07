#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-cartoons/panel/main.yaml}"
python -m ophthalmotwin_gp.studio glaze --config "${CONFIG}" "${@:2}"
