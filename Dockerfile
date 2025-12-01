FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV PATH="/root/miniconda3/bin:${PATH}"

WORKDIR /app

RUN conda create -n omnivla python=3.10 -c conda-forge --override-channels -y

SHELL ["conda", "run", "-n", "omnivla", "/bin/bash", "-c"]

RUN pip install numpy==1.26.4

RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
     --index-url https://download.pytorch.org/whl/cu118

RUN pip install packaging ninja psutil
RUN pip install "flash-attn==2.5.5" --no-build-isolation

COPY . /app/
RUN pip install -e .

RUN echo "source /root/miniconda3/etc/profile.d/conda.sh && conda activate omnivla" >> ~/.bashrc
