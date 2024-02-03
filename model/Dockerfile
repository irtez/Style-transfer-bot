FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04

# Run update and install miniconda for managing packages
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

# Install torch with cuda
RUN pip3 --timeout=300 --no-cache-dir install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other packages
COPY ./requirements.txt .
RUN pip3 --timeout=300 --no-cache-dir install -r requirements.txt

# Copy model files
COPY ./models /models

# Copy app files
COPY ./app /app
WORKDIR /app/
ENV PYTHONPATH=/app
RUN ls -lah /app/*

COPY ./start.sh /start.sh
RUN sed -i -e 's/\r$//' /start.sh
RUN chmod +x /start.sh

EXPOSE 80
CMD ["/start.sh"]