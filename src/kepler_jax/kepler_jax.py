# -*- coding: utf-8 -*-

__all__ = ["kepler"]

from functools import partial

import numpy as np
from jax import dtypes, lax
from jax import numpy as jnp
from jax.interpreters import ad, batching, masking, xla
from jax.lib import xla_client

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

xops = xla_client.ops


# This function exposes the primitive to user code and this is the only
# public-facing function in this module
def kepler(mean_anom, ecc):
    return _kepler_prim.bind(jnp.mod(mean_anom, 2 * np.pi), ecc)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# To support JIT compilation, we need a translation rule to convert the
# function into an XLA op. In our case this is the custom XLA op that we've
# written. We're wrapping two translation rules into one here: one for the CPU
# and one for the GPU
def _kepler_translation_rule(c, mean_anom, ecc, *, platform="cpu"):
    # The inputs have "shapes" that provide both the shape and the dtype
    mean_anom_shape = c.get_shape(mean_anom)

    # Extract the dtype and shape
    dtype = mean_anom_shape.element_type()
    dims = mean_anom_shape.dimensions()

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # The inputs and outputs all have the same shape so let's predefine this
    # specification
    shape = xla_client.Shape.array_shape(
        np.dtype(dtype), dims, tuple(range(len(dims) - 1, -1, -1))
    )

    # We dispatch a different call depending on the dtype
    if dtype == np.float32:
        op_name = platform.encode() + b"_kepler_f32"
    elif dtype == np.float64:
        op_name = platform.encode() + b"_kepler_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return xops.CustomCallWithLayout(
            c,
            op_name,
            # The inputs:
            operands=(xops.ConstantLiteral(c, size), mean_anom, ecc),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                shape,
                shape,
            ),
            # The output shapes:
            shape_with_layout=xla_client.Shape.tuple_shape((shape, shape)),
        )

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'kepler_jax' module was not compiled with CUDA support"
            )

        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_kepler_descriptor(size)

        return xops.CustomCallWithLayout(
            c,
            op_name,
            operands=(mean_anom, ecc),
            operand_shapes_with_layout=(shape, shape),
            shape_with_layout=xla_client.Shape.tuple_shape((shape, shape)),
            opaque=opaque,
        )

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************
def _kepler_jvp(args, tangents):
    mean_anom, ecc = args
    d_mean_anom, d_ecc = tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    sin_ecc_anom, cos_ecc_anom = _kepler_prim.bind(mean_anom, ecc)

    # Propagate the derivatives
    factor = 1 / (1 - ecc * cos_ecc_anom)
    d_ecc_anom = 0
    if type(d_mean_anom) is not ad.Zero:
        d_ecc_anom += d_mean_anom * factor
    if type(d_ecc) is not ad.Zero:
        d_ecc_anom += d_ecc * sin_ecc_anom * factor

    return (sin_ecc_anom, cos_ecc_anom), (
        cos_ecc_anom * d_ecc_anom,
        -sin_ecc_anom * d_ecc_anom,
    )


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************

# This decorator is copied from jax.lax since this was added recently and not
# available in older versions
def _broadcast_translate(translate):
    def _broadcast_array(array, array_shape, result_shape):
        if array_shape == result_shape:
            return array
        bcast_dims = tuple(
            range(len(result_shape) - len(array_shape), len(result_shape))
        )
        result = xops.BroadcastInDim(array, result_shape, bcast_dims)
        return result

    def _broadcasted_translation_rule(c, *args, **kwargs):
        shapes = [c.get_shape(arg).dimensions() for arg in args]
        result_shape = lax.broadcast_shapes(*shapes)
        args = [
            _broadcast_array(arg, arg_shape, result_shape)
            for arg, arg_shape in zip(args, shapes)
        ]
        return translate(c, *args, **kwargs)

    return _broadcasted_translation_rule


# Since we have multiple outputs, we need to infer the output dtype based on
# the inputs
def _kepler_result_dtype(*args, **kwargs):
    dtype = dtypes.canonicalize_dtype(args[0].dtype)
    return (dtype, dtype)


# Similarly we will infer the output shapes using broadcasting
def _kepler_shape_rule(*args, **kwargs):
    res = lax._broadcasting_shape_rule("kepler", *args, **kwargs)
    return (res, res)


# Define the primitive using the "standard_primitive" helper from jax.lax
_kepler_prim = lax.standard_primitive(
    _kepler_shape_rule,
    partial(
        lax.naryop_dtype_rule,
        _kepler_result_dtype,
        [{np.floating}, {np.floating}],
        "kepler",
    ),
    "kepler",
    multiple_results=True,
)

# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][_kepler_prim] = _broadcast_translate(
    partial(_kepler_translation_rule, platform="cpu")
)
xla.backend_specific_translations["gpu"][_kepler_prim] = _broadcast_translate(
    partial(_kepler_translation_rule, platform="gpu")
)

# Connect the JVP rule for autodiff
ad.primitive_jvps[_kepler_prim] = _kepler_jvp

# Add support for simple batching and masking
batching.defbroadcasting(_kepler_prim)
masking.defnaryop(_kepler_prim)
