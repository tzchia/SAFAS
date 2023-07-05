# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update the system & install dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tmux \
    vim-nox
RUN apt-get install -y zsh
WORKDIR /root
COPY vimrc /root/.vimrc
COPY zshrc /root/.zshrc
COPY oh-my-zsh /root/.oh-my-zsh
COPY tmux.conf /root/.tmux.conf
COPY . /app

# Install pip requirements
RUN pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt .
RUN pip install --upgrade pip
RUN python -m pip install -r requirements.txt


# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "train.py"]

# 978MB - 3.10-slim
# 878MB - 3.9-alpine
# - 3.9-alpine + 
