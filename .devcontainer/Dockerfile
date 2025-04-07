FROM python:3.12.9
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Set the working directory inside the container
WORKDIR /workspace

ENV PIP_NO_CACHE_DIR=off \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=2.1.1
# Install required tools and dependencies
RUN apt-get update && apt-get install -y \
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
    && curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Create a user with sudo privileges
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /etc/sudoers.d \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chown -R $USERNAME $POETRY_HOME

USER $USERNAME

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash \
    && az extension add --name ml

# Copy and install dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry install
