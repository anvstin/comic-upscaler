FROM python:3.12

SHELL ["/bin/bash", "-c"]

WORKDIR /comic-upscaler/
ADD ./upscaler ./upscaler
ADD ./test ./test
COPY main.py requirements.txt *.md ./

RUN pip3 install -r requirements.txt && pip3 cache purge

ENV ARGS="-d"

VOLUME /input /output

CMD micromamba run -n upscale --cwd /comic-upscaler/ python -- /comic-upscaler/main.py /input /output $ARGS