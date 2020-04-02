FROM python:3.5-slim
MAINTAINER gautigadu091@gmail.com
USER root
WORKDIR /app
ADD . /app
RUN apt update && apt install --no-install-recommends -y python3-dev  gcc build-essential
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install beautifulsoup4
RUN python -m spacy download en_core_web_sm
EXPOSE 80
ENTRYPOINT ["python", "app.py"]
