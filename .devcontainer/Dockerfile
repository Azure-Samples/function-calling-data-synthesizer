FROM python:3.12.13
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Set the working directory inside the container
WORKDIR /workspace

ENV PIP_NO_CACHE_DIR=off \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VERSION=2.2.1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install required tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    wget \
    curl \
    gcc \
    ca-certificates \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libffi-dev \
    llvm \
    xz-utils \
    git \
    libsqlite3-dev \
    pre-commit \
    && python -m pip install --no-cache-dir "poetry==$POETRY_VERSION" \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry install

# Install Azure CLI separately (optional; failures here won't break the build)
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash || true

# Create a user with sudo privileges
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /etc/sudoers.d \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Install Azure ML extension (optional; failures here won't break the build)
RUN az extension add --name ml || true
