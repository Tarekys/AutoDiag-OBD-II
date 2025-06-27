FROM python:3.10-slim

# إعداد مجلد العمل
WORKDIR /app

# نسخ الملفات
COPY . .

# تثبيت التبعيات
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# فتح البورت
EXPOSE 8000

# تشغيل التطبيق
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]

