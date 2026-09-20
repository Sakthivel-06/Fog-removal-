FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg for video decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Hugging Face runs on port 7860
EXPOSE 7860
ENV PORT=7860

# Run the real-time dehazing server
CMD ["python", "web_app/server.py", "7860"]
