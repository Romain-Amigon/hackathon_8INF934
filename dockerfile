FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    git \
    build-essential \
    python3-dev \
    # Nécessaire pour matplotlib et wordcloud
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Mise à jour de pip et installation des libs
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Modification de la commande pour lancer l'interface Streamlit
# Si votre fichier s'appelle app.py dans src/ui/
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]