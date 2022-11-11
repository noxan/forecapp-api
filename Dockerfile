FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt  /app
RUN pip3 install -r requirements.txt

COPY app.py /app

EXPOSE 80

CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app"]
