FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/
COPY app.py .

# Create data directory for ChromaDB
RUN mkdir -p data/chroma

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# HF Spaces requires port 7860
EXPOSE 7860

# Build KB on startup then launch dashboard
CMD ["python", "app.py"]