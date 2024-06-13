FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common wget

# Add deadsnakes PPA and install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev

# Install pip using get-pip.py script
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Confirm pip installation path
RUN which pip3

# Setup alternatives for Python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip $(which pip3) 1

# Set the working directory
COPY . /workspace
WORKDIR /workspace

# Install Python dependencies 用清华源
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# install 
RUN pip install https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_cluster-1.6.3%2Bpt22cu121-cp310-cp310-linux_x86_64.whl   https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_scatter-2.1.2%2Bpt22cu121-cp310-cp310-linux_x86_64.whl    https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_sparse-0.6.18%2Bpt22cu121-cp310-cp310-linux_x86_64.whl
RUN pip install torch_geometric
# Set the default command
CMD ["tail", "-f", "/dev/null"]
