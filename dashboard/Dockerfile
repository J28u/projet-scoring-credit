FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update

RUN pip3 install -r /app/requirements.txt

COPY ./dashboard /app

EXPOSE 80

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=80", "--server.address=0.0.0.0"]
