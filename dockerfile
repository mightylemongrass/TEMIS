FROM python:3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        build-essential \
        python3-dev \
        qt5-qmake \
        qtbase5-dev \
        qtbase5-dev-tools \
        libgl1-mesa-glx \
        libx11-dev \
        libxcb-xinerama0 \
        libxkbcommon-x11-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python3", "main.py"]
