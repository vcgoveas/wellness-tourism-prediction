
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for health checking
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 7860

# Add healthcheck to assist Hugging Face's ingress controller
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["./start.sh"]
