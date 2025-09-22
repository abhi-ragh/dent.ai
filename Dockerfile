FROM python:3.11-bookworm
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN python create_dummy_file.py 
EXPOSE 5000
CMD ["python","app.py"]
