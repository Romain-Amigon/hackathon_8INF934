# Image de base compatible avec ta Quadro P2000
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Éviter les interactions lors de l'installation
ENV DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système pour SQLite et Python
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier des dépendances
# On le crée juste en dessous si tu ne l'as pas
COPY requirements.txt .

# Installation des bibliothèques Python
# On force l'installation de llama-index et ses extensions
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code source
COPY . .

# Commande par défaut (peut être surchargée)
CMD ["python", "src/agent/agent_sql.py"]