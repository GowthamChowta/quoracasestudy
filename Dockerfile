FROM python:3.4-slim
MAINTANER gautigadu091@gmail.com
USER root
WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]