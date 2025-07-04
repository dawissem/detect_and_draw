# Thinker Paint Docker avec Python 3.11 et MediaPipe
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Thinker Paint Team"
LABEL description="Application de peinture avec détection de doigts MediaPipe"
LABEL version="1.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    # GUI dependencies
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    # System tools
    wget \
    curl \
    git \
    vim \
    # Camera support
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Créer utilisateur non-root
RUN useradd -m -s /bin/bash painter && \
    usermod -a -G video painter

# Créer répertoire de travail
WORKDIR /app

# Copier requirements en premier (pour cache Docker)
COPY requirements_docker.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Créer les dossiers nécessaires
RUN mkdir -p /app/peintures /app/captures /app/metadata /app/auto_saves /app/logs && \
    chown -R painter:painter /app

# Copier le code de l'application
COPY . .

# Donner les permissions
RUN chown -R painter:painter /app && \
    chmod +x /app/*.py

# Changer vers l'utilisateur non-root
USER painter

# Port d'exposition (si interface web ajoutée plus tard)
EXPOSE 8080

# Point d'entrée par défaut
CMD ["python", "thinker_paint_docker.py"]