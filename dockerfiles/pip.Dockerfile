FROM python:3.12

SHELL ["/bin/bash", "-c"]

WORKDIR /comic-upscaler/
COPY ./upscaler/ requirements.txt *.md ./

RUN pip3 install -r requirements.txt


