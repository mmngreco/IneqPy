FROM alpine:3.11

# This hack is widely applied to avoid python printing issues in docker containers.
# See: https://github.com/Docker-Hub-frolvlad/docker-alpine-python3/pull/13
ENV PYTHONUNBUFFERED=1

RUN echo -e "**** install Python ****"
RUN apk add --no-cache python3 python3-dev py3-virtualenv \
    && ln -sf python3 /usr/bin/python

RUN echo -e "\n\n============ install pip ============"
RUN python3 -m ensurepip                                      \
    && pip3 install --no-cache --upgrade pip setuptools wheel \
    && ln -sf pip3 /usr/bin/pip

RUN echo -e "\n\n============ install dev-tools ============"
RUN apk add --no-cache tmux zsh curl wget git neovim bash            \
   && git clone https://github.com/ohmyzsh/ohmyzsh.git ~/.oh-my-zsh  \
   && cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

RUN echo -e "\n\n============ install pyenv ============"       \
    && git clone https://github.com/pyenv/pyenv.git ~/.pyenv    \
    && echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc      \
    && echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc   \
    && echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.zshrc
# Why of below code: https://github.com/pyenv/pyenv/wiki
RUN apk add --no-cache \
    git                \
    bash               \
    build-base         \
    libffi-dev         \
    openssl-dev        \
    bzip2-dev          \
    zlib-dev           \
    readline-dev       \
    sqlite-dev
RUN apk add linux-headers


RUN echo -e "\n\n============ create git directory ============"
RUN mkdir -p /root/git
WORKDIR /root/git

RUN echo -e "\n\n============ Cloning projects ============"
RUN git clone https://github.com/mmngreco/ineqpy /root/git/ineqpy
RUN python -m venv ineqpy/venv/
RUN . ineqpy/venv/bin/activate
RUN pip install -e ./ineqpy  --no-cache

CMD /bin/zsh
