FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /atelier

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ophthalmotwin_gp ./ophthalmotwin_gp
COPY cartoons ./cartoons
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir .

ENTRYPOINT ["ophthalmotwin-gp"]
CMD ["--help"]
