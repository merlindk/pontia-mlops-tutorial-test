FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /code/app

ENV MODEL_URI="models:/test_model@champion"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
