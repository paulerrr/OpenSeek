FROM python:3.12-slim

# Chrome + Xvfb for headless browser automation
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/chromium /usr/bin/google-chrome

WORKDIR /app

COPY pyproject.toml ./
# Install openseek deps + server deps
RUN pip install --no-cache-dir \
    beautifulsoup4 \
    zendriver \
    inscriptis \
    pyvirtualdisplay \
    fastapi \
    uvicorn

COPY openseek/ ./openseek/
COPY server.py ./

ENV OPENSEEK_PORT=8888
EXPOSE 8888

CMD ["python", "server.py"]
