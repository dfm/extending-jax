FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git \
      python3-pip \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

RUN python3 -m pip install -U pip && \
    python3 -m pip install pytest && \
    python3 -m pip install -U jax jaxlib==0.1.57+cuda102 -f https://storage.googleapis.com/jax-releases/jax_releases.html

RUN cd /github/workspace && \
    python3 -m pip install . && \
    python3 -c 'import kepler_jax;print(kepler_jax.__version__)'
