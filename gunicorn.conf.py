# Access log - records incoming HTTP requests
# accesslog = "logs/gunicorn.access.log"
# Error log - records Gunicorn server goings-on
# errorlog = "logs/gunicorn.error.log"
# Whether to send Django output to the error log
# capture_output = True
# How verbose the Gunicorn error logs should be
# loglevel = "info"

wsgi_app = "main:app"

# Workers
workers = 2

graceful_timeout = 360
timeout = 360

worker_class = "uvicorn.workers.UvicornWorker"
