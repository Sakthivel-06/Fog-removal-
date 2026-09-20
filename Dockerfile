FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg for video decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (filtered by .dockerignore)
COPY . .

EXPOSE 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

CMD ["python", "web_app/server.py", "7860"]
