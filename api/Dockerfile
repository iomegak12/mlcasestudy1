FROM python:3.11-slim

LABEL author="RAMKUMAR JD"
LABEL company="INTSOL"
LABEL version="0.0.1"

WORKDIR /app

COPY requirements.txt .

RUN pip install -r ./requirements.txt && rm ./requirements.txt

EXPOSE 80

COPY . /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]