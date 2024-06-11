### 1. Get Linux
FROM debian:bullseye-slim

# Allow statements and log messages
ENV PYTHON123 True

# Port 8080
EXPOSE 8080

# Copy the requirements file into the container
COPY . /app
WORKDIR /app

### 2. Get Java via the package manager
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        bash \
        unzip \
        curl \
        openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install Python, PIP, and dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        python3-dev \
        python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --upgrade pip \
    && pip3 install torch \
    && pip3 install spacy==3.7.5 \
    && python -m spacy download en_core_web_sm \
    && rm -rf /var/lib/apt/lists/*

# Install the Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point for the container
CMD streamlit run --server.port 8080 --server.enableCORS false app.py