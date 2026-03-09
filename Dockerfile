FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src

COPY requirements.txt /app/requirements.txt

# Install CPU-only PyTorch and other dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app