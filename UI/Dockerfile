FROM python:3.9-slim

WORKDIR /app

COPY . /app

# requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

# Streamlit/HTML
CMD ["streamlit", "run", "Fault-Dashboard.py", "--server.port=10000", "--server.address=0.0.0.0"]