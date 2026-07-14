FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["streamlit", "run", "UI/Fault-Dashboard.py", "--server.port=10000", "--server.address=0.0.0.0"]
