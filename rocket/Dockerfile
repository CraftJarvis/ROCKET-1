FROM scratch
RUN echo $SSH_KEY

# FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# 安装系统依赖
# RUN apt-get update && \
#     apt-get install -y \
#     wget \
#     git \
#     gnutls-bin \
#     openssh-client \
#     libghc-x11-dev \
#     gcc-multilib \
#     g++-multilib \
#     libglew-dev \
#     libosmesa6-dev \
#     libgl1-mesa-glx \
#     libglfw3 \
#     xvfb \
#     mesa-utils \
#     libegl1-mesa \
#     libgl1-mesa-dev \
#     libglu1-mesa-dev \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     unzip 

# RUN useradd -m -u 1000 user
# USER user
# ENV PATH="/home/user/.local/bin:$PATH"

# ARG SSH_KEY
# ARG SSH_KEY_PASSPHRASE

# RUN --mount=type=secret,id=SSH_KEY,mode=0444,required=true \
#     mkdir -p /home/user/.ssh && \
#     chmod 0700 /home/user/.ssh && \
#     ssh-keyscan github.com >> /home/user/.ssh/known_hosts && \
#     echo "$(cat SSH_KEY)" > /home/user/.ssh/id_rsa && \
#     chmod 600 /home/user/.ssh/id_rsa

# # 准备资源文件
# WORKDIR /app
# COPY ./segment-anything-2-real-time.zip /app/segment-anything-2-real-time.zip
# COPY ./rocket-weights-110k-ema.ckpt /app/rocket-weights-110k-ema.ckpt
# COPY ./SAMs.zip /app/SAMs.zip
# COPY ./MCP-Reborn.zip /app/MCP-Reborn.zip
# RUN git clone git@github.com:CraftJarvis/Mark2.git && \ 
#     git clone git@github.com:karpathy/minGPT.git && \
#     unzip /app/MCP-Reborn.zip -d /app/Mark2/jarvis/stark_tech/ && \
#     unzip /app/segment-anything-2-real-time.zip -d /app/ && \
#     unzip /app/SAMs.zip -d /app/

# # 配置 conda 环境并安装 pip 依赖
# WORKDIR /app
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     -O /tmp/miniconda.sh && \
#     bash /tmp/miniconda.sh -b -p /app/conda 
# ENV PATH /app/conda/bin:$PATH
# RUN conda init bash 
# RUN conda create -n rocket python=3.10 -y && \
#     conda run -n rocket conda install --channel=conda-forge openjdk=8 -y 

# # 激活 Conda 环境，并保持激活状态
# SHELL ["/bin/bash", "-c"]
# RUN echo "conda activate rocket" >> ~/.bashrc

# WORKDIR /app
# RUN cd /app/segment-anything-2-real-time && \
#     TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" conda run -n rocket pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .

# WORKDIR /app
# RUN cd minGPT && \
#     conda run -n rocket pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .

# WORKDIR /app
# RUN cd Mark2 && \
#     git checkout main && \
#     git pull && \
#     conda run -n rocket pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e . 

# # WORKDIR /app/Mark2/jarvis/assembly
# # CMD [ "python", "eval_rocket_molmo.py", "--port", "7860", "--rocket-path", "/app/rocket-weights-110k-ema.ckpt" \
# #              "--sam-path", "/app/SAMs", "--molmo-id", "molmo-72b-0924", "--molmo-url", "http://172.17.30.127:8000/v1" ]