FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt  /app
RUN pip3 install -r requirements.txt

COPY app.py /app

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
