FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY templates/ templates/
COPY classifier.py .
RUN python classifier.py
COPY start.sh .
RUN chmod +x start.sh
EXPOSE 8001 5000
CMD ["./start.sh"]