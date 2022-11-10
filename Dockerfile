FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt  /app
RUN pip3 install -r requirements.txt

COPY app.py /app

EXPOSE 8000

CMD ["gunicorn app:app"]
