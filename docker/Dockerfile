# This Dockerfile is based on https://github.com/ikeyasu/docker-reinforcement-learning

# To use cuda9.2 container, you need to install nvidia-driver >= 396.26
# See https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements
FROM syuntoku/rl_ws:latest
MAINTAINER syuntoku14 <syuntoku14@gmail.com>

RUN git clone git@github.com:syuntoku14/pytorch-rl-il.git
RUN cd pytorch-rl-il && pip install -e .

CMD ["bash"]
WORKDIR /root