FROM ubuntu:20.04

# Ustawienia lokalizacyjne
ENV TZ=Europe/Warsaw
ENV DEBIAN_FRONTEND=noninteractive

# Instalacja zależności systemowych + Python 3.8
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-venv \
    python3-dev \
    build-essential \
    wget \
    curl \
    vim \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Ustawienie python jako domyślny
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Instalacja bibliotek Pythonowych (bez wsparcia GPU)
RUN pip install --upgrade pip && \
    pip install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install numpy scipy matplotlib opencv-python scikit-image Cython tensorboardX

# Domyślna ścieżka robocza
WORKDIR /app

CMD [ "python" ]
