FROM python:3.11-bookworm
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
CMD ["python","app.py"]
