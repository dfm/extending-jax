# -*- coding: utf-8 -*-

__all__ = ["kepler"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

# This function exposes the primitive to user code and this is the only
# public-facing function in this module


def kepler(mean_anom, ecc):
    # We're going to apply array broadcasting here since the logic of our op
    # is much simpler if we require the inputs to all have the same shapes
    mean_anom_, ecc_ = jnp.broadcast_arrays(mean_anom, ecc)

    # Then we need to wrap into the range [0, 2*pi)
    M_mod = jnp.mod(mean_anom_, 2 * np.pi)

    return _kepler_prim.bind(M_mod, ecc_)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _kepler_abstract(mean_anom, ecc):
    shape = mean_anom.shape
    dtype = dtypes.canonicalize_dtype(mean_anom.dtype)
    assert dtypes.canonicalize_dtype(ecc.dtype) == dtype
    assert ecc.shape == shape
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _kepler_lowering(ctx, mean_anom, ecc, *, platform="cpu"):

    # Checking that input types and shape agree
    assert mean_anom.type == ecc.type

    # Extract the numpy type of the inputs
    mean_anom_aval, _ = ctx.avals_in
    np_dtype = np.dtype(mean_anom_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(mean_anom.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # We dispatch a different call depending on the dtype
    if np_dtype == np.float32:
        op_name = platform + "_kepler_f32"
    elif np_dtype == np.float64:
        op_name = platform + "_kepler_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return custom_call(
            op_name,
            # Output types
            result_types=[dtype, dtype],
            # The inputs:
            operands=[mlir.ir_constant(size), mean_anom, ecc],
            # Layout specification:
            operand_layouts=[(), layout, layout],
            result_layouts=[layout, layout]
        ).results

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'kepler_jax' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_kepler_descriptor(size)

        return custom_call(
            op_name,
            # Output types
            result_types=[dtype, dtype],
            # The inputs:
            operands=[mean_anom, ecc],
            # Layout specification:
            operand_layouts=[layout, layout],
            result_layouts=[layout, layout],
            # GPU specific additional data
            backend_config=opaque
        ).results

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# Here we define the differentiation rules using a JVP derived using implicit
# differentiation of Kepler's equation:
#
#  M = E - e * sin(E)
#  -> dM = dE * (1 - e * cos(E)) - de * sin(E)
#  -> dE/dM = 1 / (1 - e * cos(E))  and  de/dM = sin(E) / (1 - e * cos(E))
#
# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
def _kepler_jvp(args, tangents):
    mean_anom, ecc = args
    d_mean_anom, d_ecc = tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    sin_ecc_anom, cos_ecc_anom = _kepler_prim.bind(mean_anom, ecc)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # Propagate the derivatives
    d_ecc_anom = (
        zero_tangent(d_mean_anom, mean_anom)
        + zero_tangent(d_ecc, ecc) * sin_ecc_anom
    ) / (1 - ecc * cos_ecc_anom)

    return (sin_ecc_anom, cos_ecc_anom), (
        cos_ecc_anom * d_ecc_anom,
        -sin_ecc_anom * d_ecc_anom,
    )


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _kepler_batch(args, axes):
    assert axes[0] == axes[1]
    return kepler(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_kepler_prim = core.Primitive("kepler")
_kepler_prim.multiple_results = True
_kepler_prim.def_impl(partial(xla.apply_primitive, _kepler_prim))
_kepler_prim.def_abstract_eval(_kepler_abstract)

# Connect the XLA translation rules for JIT compilation
for platform in ["cpu", "gpu"]:
    mlir.register_lowering(
        _kepler_prim,
        partial(_kepler_lowering, platform=platform),
        platform=platform)

# Connect the JVP and batching rules
ad.primitive_jvps[_kepler_prim] = _kepler_jvp
batching.primitive_batchers[_kepler_prim] = _kepler_batch
