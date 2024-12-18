FROM mambaorg/micromamba:latest

SHELL ["/bin/bash", "-c"]

WORKDIR /comic-upscaler/
COPY ./upscaler/ environment.yml *.md ./


RUN micromamba env create -f ./environment.yml -y && \
    micromamba clean --all


