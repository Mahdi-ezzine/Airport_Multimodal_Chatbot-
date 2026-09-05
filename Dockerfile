# Base on a slim Python image rather than a full distribution, to keep the image small
FROM python:3.11-slim

# Install the system packages the Python libraries depend on
  # ffmpeg is REQUIRED: Whisper shells out to it to decode audio files
  # libgl1 and libglib2.0-0 are required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Work inside a dedicated directory
WORKDIR /app

# Install CPU-only PyTorch first, from the dedicated CPU index
  # The image must run on machines without an NVIDIA GPU, and the CUDA build
  # is around 2 GB larger while being unusable there
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# Copy the requirements produced by Step 0.4 and install them
  # torch is commented out in that file, so the CPU build above is preserved
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model weights at BUILD time so the container is self-contained
  # Without this the first passenger request downloads roughly 900 MB, which makes
  # the container non-reproducible and unusable offline
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2'); \
import whisper; whisper.load_model('base'); \
print('All model weights cached in the image.')"

# Copy the application modules and the knowledge base
COPY airport_config.py kb_store.py perception.py fusion.py app.py ./
COPY airport_kb.json ./

# Run as a non-root user, so a compromised container cannot act as root on the host
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Document the port the application listens on
EXPOSE 8501

# Let Docker report the container as unhealthy if Streamlit stops responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Start the application, listening on every interface so the port can be published
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
