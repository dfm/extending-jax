#!/bin/sh -l

cd /github/workspace
python3 -m pip install .
python3 -c 'import kepler_jax;print(kepler_jax.__version__)'
