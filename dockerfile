FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

WORKDIR /app/src

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]