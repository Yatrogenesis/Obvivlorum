# Dockerfile para Obvivlorum Sandbox
FROM python:3.11-slim-bullseye

# Metadatos
LABEL maintainer="Francisco Molina <pako.molina@gmail.com>"
LABEL version="2.0"
LABEL description="AI Symbiote Obvivlorum - Secure Sandbox Environment"

# Variables de construcción
ARG USER_ID=1000
ARG GROUP_ID=1000

# Crear usuario no-root
RUN groupadd -g ${GROUP_ID} obvivlorum && \
    useradd -m -u ${USER_ID} -g obvivlorum -s /bin/bash obvivlorum

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    git \
    curl \
    wget \
    vim \
    netcat \
    net-tools \
    iputils-ping \
    dnsutils \
    tcpdump \
    nmap \
    nikto \
    dirb \
    sqlmap \
    hashcat \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código (como usuario no-root)
COPY --chown=obvivlorum:obvivlorum . /app

# Crear directorios necesarios
RUN mkdir -p /app/logs /app/data /app/.cache && \
    chown -R obvivlorum:obvivlorum /app

# Script de entrada seguro
COPY --chown=obvivlorum:obvivlorum docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Cambiar a usuario no-root
USER obvivlorum

# Exponer puerto para interfaces web
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; import os; sys.exit(0 if os.path.exists('/app/ai_symbiote.py') else 1)"

# Entrada
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "ai_symbiote.py", "--sandbox"]