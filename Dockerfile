# syntax = docker/dockerfile:experimental
#
# This file can build images for cpu and gpu env. By default it builds image for CPU.
# Use following option to build image for cuda/GPU: --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# Here is complete command for GPU/cuda - 
# $ DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 -t torchserve:latest .
#
# Following comments have been shamelessly copied from https://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/


ARG BASE_IMAGE=nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04
ARG CUDA_VERSION=cu111

FROM ${BASE_IMAGE} AS compile-image
ARG BASE_IMAGE=ubuntu:18.04
ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    rm /etc/apt/sources.list.d/* && \
    apt-get clean && \
    apt-get update && \
    # apt-get install wget -y && \
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    # dpkg -i cuda-keyring_1.0-1_all.deb \
    #apt --fix-broken -y install && \
    apt remove python-pip  python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3.8-venv \
    python3-venv \
    openjdk-11-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py


RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \ 
    && update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3.8 1

RUN python3.8 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN python -m pip install -U pip setuptools

# This is only useful for cuda env
RUN export USE_CUDA=1

ARG CUDA_VERSION=""

RUN TORCH_VER=$(curl --silent --location https://pypi.org/pypi/torch/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") && \
    TORCH_VISION_VER=$(curl --silent --location https://pypi.org/pypi/torchvision/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") && \
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        # Install CUDA version specific binary when CUDA version is specified as a build arg
        if [ "$CUDA_VERSION" ]; then \
            python -m pip install --no-cache-dir torch==$TORCH_VER+$CUDA_VERSION torchvision==$TORCH_VISION_VER+$CUDA_VERSION -f https://download.pytorch.org/whl/torch_stable.html; \
        # Install the binary with the latest CUDA version support
        else \
            python -m pip install --no-cache-dir torch torchvision; \
        fi; \
        python -m pip install --no-cache-dir -r https://raw.githubusercontent.com/pytorch/serve/master/requirements/common.txt; \
    # Install the CPU binary
    else \
        python -m pip install --no-cache-dir torch==$TORCH_VER+cpu torchvision==$TORCH_VISION_VER+cpu -f https://download.pytorch.org/whl/torch_stable.html; \
    fi
RUN python -m pip install -U setuptools && python -m pip install --no-cache-dir captum torchtext torchserve torch-model-archiver

# Final image for production
FROM ${BASE_IMAGE} AS runtime-image

ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,target=/var/cache/apt \
    rm /etc/apt/sources.list.d/* && \
    apt-get clean && \
    # apt-key del 7fa2af80 && \
    apt-get update && \
    # apt-get install wget -y && \
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    # dpkg -i cuda-keyring_1.0-1_all.deb \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3.8 \
    python3.8-distutils \
    python3.8-dev \
    openjdk-11-jre-headless \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY --chown=model-server --from=compile-image /home/venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
COPY ./important/* .

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server

RUN pip install gdown nvgpu grpcio protobuf grpcio-tools \
    && gdown 1oHQZ3qX_ZopgstOiqrfPTVX35I5GMw9d

RUN mkdir /home/model-server/model-store && \
    mv schp-atr.mar /home/model-server/model-store/schp-atr.mar && \
    chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081 8082 7070 7071

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp

ENTRYPOINT [ "tail -f /dev/null" ]
# ENTRYPOINT torchserve --start && \
#            python -m grpc_tools.protoc --proto_path=./proto/ \ 
#            --python_out=ts_scripts --grpc_python_out=ts_scripts \ 
#            ./proto/inference.proto ./proto/management.proto
# ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
# CMD ["serve"]