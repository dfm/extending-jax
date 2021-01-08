# -*- coding: utf-8 -*-

__all__ = ["kepler"]

from functools import partial

import numpy as np
from jax import core
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, xla
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


# Define the primitive
_kepler_prim = core.Primitive("kepler")

# It will have multiple outputs (don't set this if your op only has one)
_kepler_prim.multiple_results = True

# We don't have any implementation besides the XLA version. If you had a
# simpler implementation (in numpy, for example) you could use that here
_kepler_prim.def_impl(partial(xla.apply_primitive, _kepler_prim))


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# To support JIT compilation, we need 2 pieces:

# (1) An abstract evaluation function to compute the shape and dtype of the
#     outputs given the shape and dtype of the expected inputs
@_kepler_prim.def_abstract_eval
def _kepler_abstract_eval(mean_anom, ecc):
    if mean_anom.dtype != ecc.dtype:
        raise ValueError("Data type precision of inputs must match")
    if mean_anom.shape != ecc.shape:
        raise ValueError("Dimensions of inputs must match")
    return (
        ShapedArray(mean_anom.shape, mean_anom.dtype),
        ShapedArray(mean_anom.shape, mean_anom.dtype),
    )


# (2) A translation rule to convert the function into an XLA op. In our case
#     this is the custom XLA op that we've written. We're wrapping two
#     translation rules into one here: one for the CPU and one for the GPU
def _kepler_translation_rule(c, mean_anom, ecc, *, platform="cpu"):
    # The inputs have "shapes" that provide both the shape and the dtype
    mean_anom_shape = c.get_shape(mean_anom)
    ecc_shape = c.get_shape(ecc)

    # Extract the dtype and shape
    dtype = mean_anom_shape.element_type()
    dims = mean_anom_shape.dimensions()

    # Make sure that these match as expected
    if ecc_shape.element_type() != dtype:
        raise ValueError("Data type precision of inputs must match")
    if ecc_shape.dimensions() != dims:
        raise ValueError("Dimensions of inputs must match")

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # The inputs and outputs all have the same shape so let's predefine this
    # specification
    shape = xla_client.Shape.array_shape(
        np.dtype(dtype), dims, tuple(range(len(dims) - 1, -1, -1))
    )

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we dispatch a different call depending on the dtype
        if dtype == np.float32:
            op_name = b"cpu_kepler_f32"
        elif dtype == np.float64:
            op_name = b"cpu_kepler_f64"
        else:
            raise NotImplementedError(f"Unsupported dtype {dtype}")

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

        # On the GPU, we do things a little differently and encapsulate both
        # the dtype and the dimension using the 'opaque' parameter. This isn't
        # strictly necessary. We could, for example, have different GPU ops
        # for each dtype like we did for the CPU version, but we would provide
        # provide the dimensions using the 'opaque' parameter anyways so it
        # doesn't hurt to do it like this.
        if dtype == np.float32:
            xla_dtype = gpu_ops.Type.float32
        elif dtype == np.float64:
            xla_dtype = gpu_ops.Type.float64
        else:
            raise NotImplementedError(f"Unsupported dtype {dtype}")

        # Build the problem descriptor to be passed using 'opaque' below
        opaque = gpu_ops.build_kepler_descriptor(xla_dtype, size)

        # Describe the custom call with nearly the same inputs as above
        return xops.CustomCallWithLayout(
            c,
            b"gpu_kepler",
            operands=(mean_anom, ecc),
            operand_shapes_with_layout=(shape, shape),
            shape_with_layout=xla_client.Shape.tuple_shape((shape, shape)),
            opaque=opaque,
        )

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )


xla.backend_specific_translations["cpu"][_kepler_prim] = partial(
    _kepler_translation_rule, platform="cpu"
)
xla.backend_specific_translations["gpu"][_kepler_prim] = partial(
    _kepler_translation_rule, platform="gpu"
)


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************
def _kepler_jvp(args, tangents):
    mean_anom, ecc = args
    d_mean_anom, d_ecc = tangents

    # We use bind here instead of the Kepler function because we don't want
    # to mod the mean anomaly again
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


ad.primitive_jvps[_kepler_prim] = _kepler_jvp


# ***********************************
# *  SUPPORT FOR BATCHING VIA VMAP  *
# ***********************************
def _kepler_batch(args, axes):
    assert axes[0] == axes[1]
    return kepler(*args), axes


batching.primitive_batchers[_kepler_prim] = _kepler_batch
