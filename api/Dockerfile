FROM python:3.9.12

COPY ./requirements.txt /code/requirements.txt

WORKDIR /code

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./api/data /code/data
COPY ./api/models /code/models
COPY ./api/app /code/app

EXPOSE 80

CMD ["uvicorn", "app.main:app_fastapi", "--host", "0.0.0.0", "--port", "80"]
