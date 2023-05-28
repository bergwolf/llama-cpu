# Dockerfile to deploy a llama-cpp container with conda-ready environments 

# docker pull continuumio/miniconda3:latest

ARG TAG=latest
FROM continuumio/miniconda3:$TAG AS base

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        locales \
        openssh-server \
        nano \
    && rm -rf /var/lib/apt/lists/*

# Setting up locales
RUN locale-gen en_US.UTF-8

# SSH exposition
RUN rm -f /etc/ssh/ssh_host_ed25519_key
RUN ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -P ""
RUN passwd -d root
COPY id_rsa.pub /root/.ssh/authorized_keys

# Copy llama-cpu
COPY llama-cpu /llama-cpu

# Updating conda to the latest version
RUN conda update conda -y

# Create virtalenv
RUN conda create -n llama -y python=3.10.6

# conda init bash for $user
RUN conda init bash

RUN cd /llama-cpu; pip install -r requirements.txt

# COPY entrypoint.sh /usr/bin/entrypoint
# RUN chmod 755 /usr/bin/entrypoint
# ENTRYPOINT ["/usr/bin/entrypoint"]

# Preparing for login
#ENV HOME /home/llama-cpp-user
#WORKDIR ${HOME}/llama.cpp
#USER llama-cpp-user

FROM scratch

COPY --from=base / /
ENV LANG en_US.UTF-8

EXPOSE 22/tcp
RUN service ssh start

CMD ["/bin/bash"]
