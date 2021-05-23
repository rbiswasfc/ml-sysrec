FROM python:3.8-slim
WORKDIR /sys_rec
COPY requirements.txt .
RUN pip install -r requirements.txt 
COPY . .
CMD ["python", "index.py"]