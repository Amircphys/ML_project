FROM python:3.11-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY s3cfg /root/.s3cfg
COPY .env ./
COPY main.py ./main.py
COPY transformer.pkl ./transformer.pkl

WORKDIR .

CMD ["python", "main.py"]