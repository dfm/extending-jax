#!/usr/bin/env python

import os,codecs
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# -- read long description --
HERE = os.path.dirname(os.path.realpath(__file__))
def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()

# -- add source files --
extensions = [
    CUDAExtension(
        "kepler_jax.cpu_ops",
        ["lib/cpu_ops.cc"],
    ),
]
if os.environ.get("KEPLER_JAX_CUDA", "no").lower() == "yes":
    extensions.append(
        CUDAExtension(
            "kepler_jax.gpu_ops",
            [
                "lib/gpu_ops.cc",
                "lib/kernels.cc.cu",
            ],
        )
    )

# -- run python make --
setup(
    name="kepler_jax",
    author="Dan Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    url="https://github.com/dfm/extending-jax",
    license="MIT",
    description=(
        "A simple demonstration of how you can extend JAX with custom C++ and "
        "CUDA ops"
    ),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["jax", "jaxlib"],
    extras_require={"test": "pytest"},
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)
