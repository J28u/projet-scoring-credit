FROM python:3.10

COPY ./requirements.txt /code/requirements.txt

WORKDIR /code

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./api/data /code/data
COPY ./api/models /code/models
COPY ./api/app /code/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
