FROM mambaorg/micromamba:latest

SHELL ["/bin/bash", "-c"]

WORKDIR /comic-upscaler/
ADD ./upscaler ./upscaler
ADD ./test ./test
COPY main.py environment.yml *.md ./


RUN micromamba env create -f ./environment.yml -y && \
    micromamba clean --all

ENV ARGS="-d"

VOLUME /input /output

CMD micromamba run -n upscale --cwd /comic-upscaler/ python -- /comic-upscaler/main.py /input /output $ARGS

