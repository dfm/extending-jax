#!/bin/sh -l

cd /github/workspace
KEPLER_JAX_CUDA=yes python3 -m pip install -v .
python3 -c 'import kepler_jax;print(kepler_jax.__version__)'
python3 -c 'import kepler_jax.gpu_ops'
