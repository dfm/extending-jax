> [!WARNING]  
> The JAX documentation now a supported interface for interfacing with C++ and
> CUDA libraries. Check out [the official tutorial](https://jax.readthedocs.io/en/latest/ffi.html),
> which should be preferred to the methods described here.

# Extending JAX with custom C++ and CUDA code

[![Tests](https://github.com/dfm/extending-jax/workflows/Tests/badge.svg)](https://github.com/dfm/extending-jax/actions?query=workflow%3ATests)

This repository is meant as a tutorial demonstrating the infrastructure required
to provide custom ops in JAX when you have an existing implementation in C++
and, optionally, CUDA. I originally wanted to write this as a blog post, but
there's enough boilerplate code that I ended up deciding that it made more sense
to just share it as a repo with the tutorial in the README, so here we are!

The motivation for this is that in my work I want to use libraries like JAX to
fit models to data in astrophysics. In these models, there is often at least one
part of the model specification that is physically motivated and while there are
generally existing implementations of these model elements, it is often
inefficient or impractical to re-implement these as a high-level JAX function.
Instead, I want to expose a well-tested and optimized implementation in C
directly to JAX. In my work, this often includes things like iterative
algorithms or special functions that are not well suited to implementation using
JAX directly.

So, as part of updating my [exoplanet](https://docs.exoplanet.codes) library to
interface with JAX, I had to learn what infrastructure was required to support
this use case, and since I couldn't find a tutorial that covered all the pieces
that I needed in one place, I wanted to put this together. Pretty much
everything that I'll talk about is covered in more detail somewhere else (even
if that somewhere is just a comment in some source code), but hopefully this
summary can point you in the right direction if you have a use case like this.

**A warning**: I'm writing this in January 2021 (most recent update November 2023; see
github for the full revision history) and much of what I'm talking about is based on
essentially undocumented APIs that are likely to change.
Furthermore, I'm not affiliated with the JAX project and I'm far from an expert
so I'm sure there are wrong things that I say. I'll try to update this if I
notice things changing or if I learn of issues, but no promises! So, MIT license
and all that: use at your own risk.

## Related reading

As I mentioned previously, this tutorial is built on a lot of existing
literature and I won't reproduce all the details of those documents here, so I
wanted to start by listing the key resources that I found useful:

1. The [How primitives work][jax-primitives] tutorial in the JAX documentation
   includes almost all the details about how to expose a custom op to JAX and
   spending some quality time with that tutorial is not wasted time. The only
   thing missing from that document is a description of how to use the XLA
   CustomCall interface.

2. Which brings us to the [XLA custom calls][xla-custom] documentation. This
   page is pretty telegraphic, but it includes a description of the interface
   that your custom call functions need to support. In particular, this is where
   the differences in interface between the CPU and GPU are described, including
   things like the "opaque" parameter and how multiple outputs are handled.

3. I originally learned how to write the pybind11 interface for an XLA custom
   call from the [danieljtait/jax_xla_adventures][xla-adventures] repository by
   Dan Tait on GitHub. Again, this doesn't include very many details, but that's
   really a benefit here because it really distills the infrastructure to a
   place where I could understand what was going on.

4. Finally, much of what I know about this topic, I learned from spelunking in
   the [jaxlib source code][jaxlib] on GitHub. That code is pretty readable and
   includes good comments most of the time so that's a good place to look if you
   get stuck since folks there might have already faced the issue.

## What is an "op"

In frameworks like JAX (or Theano, or TensorFlow, or PyTorch, to name a few),
models are defined as a collection of operations or "ops" that can be chained,
fused, or differentiated in clever ways. For our purposes, an op defines a
function that knows:

1. how the input and output parameter shapes and types are related,
2. how to compute the output from a set of inputs, and
3. how to propagate derivatives using the chain rule.

There are a lot of choices about where you draw the lines around a single op and
there will be tradeoffs in terms of performance, generality, ease of use, and
other factors when making these decisions. In my experience, it is often best to
define the minimal scope ops and then allow your framework of choice to combine
it efficiently with the rest of your model, but there will always be counter
examples.

## Our example application: solving Kepler's equation

In this section I'll describe the application presented in this project. Feel
free to skip this if you just want to get to the technical details.

This project exposes a single jit-able and differentiable JAX operation to solve
[Kepler's equation][keplers-equation], a tool that is used for computing
gravitational orbits in astronomy. This is basically the "hello world" example
that I use whenever learning about something like this. For example, I have
previously written [about how to expose such an op when using Stan][stan-cpp].
The implementation used in that post and the one used here are not meant to be
the most robust or efficient, but it is relatively simple and it exposes some of
the interesting issues that one might face when writing custom JAX ops. If
you're interested in the mathematical details, take a look at [my blog
post][stan-cpp], but the key point for now is that this operation involves
solving a transcendental equation, and in this tutorial we'll use a simple
iterative method that you'll find in the [kepler.h][kepler-h] header file. Then,
the derivatives of this operation can be evaluated using implicit
differentiation. Unlike in the previously mentioned blog post, our operation
will actually return the sine and cosine of the eccentric anomaly, since that's
what most high performance versions of this function would return and because
the way XLA handles ops with multiple outputs is a little funky.

## The cost/benefit analysis

One important question to answer first is: "should I actually write a custom JAX
extension?" If you're here, you've probably already thought about that, but I
wanted to emphasize a few points to consider.

1. **Performance**: The main reason why you might want to implement a custom op
   for JAX is performance. JAX's JIT compiler can get great performance in a
   broad range of applications, but for some of the problems I work on,
   finely-tuned C++ can be much faster. In my experience, iterative algorithms,
   other special functions, or code with complicated logic are all examples of
   places where a custom op might greatly improve performance. I'm not always
   good at doing this, but it's probably worth benchmarking performance of a
   version of your code implemented directly in high-level JAX against your
   custom op.

2. **Autodiff**: One thing that is important to realize is that the extension
   that we write won't magically know how to propagate derivatives. Instead,
   we'll be required to provide a JAX interface for applying the chain rule to
   out op. In other words, if you're setting out to wrap that huge Fortran
   library that has been passed down through the generations, the payoff might
   not be as great as you hoped unless (a) the code already provides operations
   for propagating derivatives (in which case you JAX op probably won't support
   second and higher order differentiation), or (b) you can easily compute the
   differentiation rules using the algorithm that you already have (which is the
   case we have for our example application here). In my work, I try (sometimes
   unsuccessfully) to identify the minimum number and size of ops that I can get
   away with and then implement most of my models directly in JAX. In our demo
   application, for example, I could have chosen to make an XLA op generating a
   full radial velocity model, instead of just solving Kepler's equation, and
   that might (or might not) give better performance. But, the differentiation
   rules are _much_ simpler the way it is implemented.

## Summary of the relevant files

The files in this repo come in three categories:

1. In the root directory, there are the standard packaging files like a
   `pyproject.toml`. Most of this setup is pretty standard, but
   I'll highlight some unique elements in the packaging section below.

2. Next, the `src/kepler_jax` directory is a Python module with the definition
   of our JAX primitive roughly following the JAX [How primitives
   work][jax-primitives] tutorial.

3. Finally, the C++ and CUDA code implementing our XLA op live in the `lib`
   directory. The `pybind11_kernel_helpers.h` and `kernel_helpers.h` headers are
   boilerplate necessary for building in the interface. The rest of the files
   include the code specific for this implementation, but I'll describe this in
   more detail below.

## Defining an XLA custom call on the CPU

The algorithm for our example problem is is implemented in the `lib/kepler.h`
header and I won't go into details about the algorithm here, but the main point
is that this could be an implementation built on any external library that you
can call from C++ and, if you want to support GPU usage, CUDA. That header file
includes a single function `compute_eccentric_anomaly` with the following
signature:

```c++
template <typename T>
void compute_eccentric_anomaly(
   const T& mean_anom, const T& ecc, T* sin_ecc_anom, T* cos_ecc_anom
);
```

This is the function that we want to expose to JAX.

As described in the [XLA documentation][xla-custom], the signature for a CPU XLA
custom call in C++ is:

```c++
void custom_call(void* out, const void** in);
```

where, as you might expect, the elements of `in` point to the input values. So,
in our case, the inputs are an integer giving the dimension of the problem
`size`, an array with the mean anomalies `mean_anomaly`, and an array of
eccentricities `ecc`. Therefore, we might parse the input as follows:

```c++
#include <cstdint>  // int64_t

template <typename T>
void cpu_kepler(void *out, const void **in) {
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const T *mean_anom = reinterpret_cast<const T *>(in[1]);
  const T *ecc = reinterpret_cast<const T *>(in[2]);
}
```

Here we have used a template so that we can support both single and double
precision version of the op.

The output parameter is somewhat more complicated. If your op only has one
output, you would access it using

```c++
T *result = reinterpret_cast<T *>(out);
```

but when you have multiple outputs, things get a little hairy. In our example,
we have two outputs, the sine `sin_ecc_anom` and cosine `cos_ecc_anom` of the
eccentric anomaly. Therefore, our `out` parameter -- even though it looks like a
`void*` -- is actually a `void**`! Therefore, we will access the output as
follows:

```c++
template <typename T>
void cpu_kepler(void *out_tuple, const void **in) {
  // ...
  void **out = reinterpret_cast<void **>(out_tuple);
  T *sin_ecc_anom = reinterpret_cast<T *>(out[0]);
  T *cos_ecc_anom = reinterpret_cast<T *>(out[1]);
}
```

Then finally, we actually apply the op and the full implementation, which you
can find in `lib/cpu_ops.cc` is:

```c++
// lib/cpu_ops.cc
#include <cstdint>

template <typename T>
void cpu_kepler(void *out_tuple, const void **in) {
  const std::int64_t size = *reinterpret_cast<const std::int64_t *>(in[0]);
  const T *mean_anom = reinterpret_cast<const T *>(in[1]);
  const T *ecc = reinterpret_cast<const T *>(in[2]);

  void **out = reinterpret_cast<void **>(out_tuple);
  T *sin_ecc_anom = reinterpret_cast<T *>(out[0]);
  T *cos_ecc_anom = reinterpret_cast<T *>(out[1]);

  for (std::int64_t n = 0; n < size; ++n) {
    compute_eccentric_anomaly(mean_anom[n], ecc[n], sin_ecc_anom + n, cos_ecc_anom + n);
  }
}
```

and that's it!

## Building & packaging for the CPU

Now that we have an implementation of our XLA custom call target, we need to
expose it to JAX. This is done by compiling a CPython module that wraps this
function as a [`PyCapsule`][capsule] type. This can be done using pybind11,
Cython, SWIG, or the Python C API directly, but for this example we'll use
pybind11 since that's what I'm most familiar with. The [LAPACK ops in
jaxlib][jaxlib-lapack] are implemented using Cython if you'd like to see an
example of how to do that.

Another choice that I've made is to use [scikit-build-core](scikit-build-core)
and [CMake](https://cmake.org) to build the extensions. Another build option
would be to use [bazel](https://bazel.build) to compile the code, like the JAX
project, but I don't have any experience with it, so I decided to stick with
what I know. _The key point is that we're just compiling a regular old Python
module, so you can use whatever infrastructure you're familiar with!_

With these choices out of the way, the boilerplate code required to define the
interface is, using the `cpu_kepler` function defined in the previous section as
follows:

```c++
// lib/cpu_ops.cc
#include <pybind11/pybind11.h>

// If you're looking for it, this function is actually implemented in
// lib/pybind11_kernel_helpers.h
template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["cpu_kepler_f32"] = EncapsulateFunction(cpu_kepler<float>);
  dict["cpu_kepler_f64"] = EncapsulateFunction(cpu_kepler<double>);
  return dict;
}

PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }
```

In this case, we're exporting a separate function for both single and double
precision. Another option would be to pass the data type to the function and
perform the dispatch logic directly in C++, but I find it cleaner to do it like
this.

With that out of the way, the actual build routine is defined in the following
files:

- In `./pyproject.toml`, we specify that `pybind11` and `scikit-build-core` are
  required build dependencies and that we'll use `scikit-build-core` as the
  build backend.

- Then, `CMakeLists.txt` defines the build process for CMake using [pybind11's
  support for CMake builds][pybind11-cmake]. This will also, optionally, build
  the GPU ops as discussed below.

With these files in place, we can now compile our XLA custom call ops using

```bash
pip install .
```

The final thing that I wanted to reiterate in this section is that
`kepler_jax.cpu_ops` is just a regular old CPython extension module, so anything
that you already know about packaging C extensions or any other resources that
you can find on that topic can be applied. This wasn't obvious when I first
started learning about this so I definitely went down some rabbit holes that
hopefully you can avoid.

## Exposing this op as a JAX primitive

The main components that are required to now call our custom op from JAX are
well covered by the [How primitives work][jax-primitives] tutorial, so I won't
reproduce all of that here. Instead I'll summarize the key points and then
provide the missing part. If you haven't already, you should definitely read
that tutorial before getting started on this part.

In summary, we will define a `jax.core.Primitive` object with an "abstract
evaluation" rule (see `src/kepler_jax/kepler_jax.py` for all the details)
following the primitives tutorial. Then, we'll add a "translation rule" and a
"JVP rule". We're lucky in this case, and we don't need to add a "transpose
rule". JAX can actually work that out automatically, since our primitive is not
itself used in the calculation of the output tangents. This won't always be
true, and the [How primitives work][jax-primitives] tutorial includes an example
of what to do in that case.

Before defining these rules, we need to register the custom call target with
JAX. To do that, we import our compiled `cpu_ops` extension module from above
and use the `registrations` dictionary that we defined:

```python
from jax.lib import xla_client
from kepler_jax import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")
```

Then, the **lowering rule** is defined roughly as follows (the one you'll
find in the source code is a little more complicated since it supports both CPU
and GPU translation):

```python
# src/kepler_jax/kepler_jax.py
import numpy as np
from jax.interpreters import mlir
from jaxlib.mhlo_helpers import custom_call

def _kepler_lowering(ctx, mean_anom, ecc):

    # Checking that input types and shape agree
    assert mean_anom.type == ecc.type

    # Extract the numpy type of the inputs
    mean_anom_aval, ecc_aval = ctx.avals_in
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
        op_name = "cpu_kepler_f32"
    elif np_dtype == np.float64:
        op_name = "cpu_kepler_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

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

mlir.register_lowering(
        _kepler_prim,
        _kepler_lowering,
        platform="cpu")
```

There appears to be a lot going on here, but most of it is just type checking.
The main meat of it is the `custom_call` function which is a thin convenience
wrapper around the `mhlo.CustomCallOp` (documented
[here](https://www.tensorflow.org/mlir/hlo_ops#mhlocustom_call_mlirmhlocustomcallop)).
Here's a summary of its arguments:

- The first argument is the name that you gave your `PyCapsule`
  in the `registrations` dictionary in `lib/cpu_ops.cc`. You can check what
  names your capsules had by looking at `cpu_ops.registrations().keys()`.

- Then, the two following arguments give the "type" of the outputs, and
  specify the input arguments (operands). In this context, a "type" is
  specified by a data type defining the size of each dimension (what I
  would normally call the shape), and the type of the array (e.g. float32).
  In this case, both our outputs have the same type/shape.

- Finally, with the last two arguments, we specify the memory layout
  of both input and output buffers.

It's worth remembering that we're expecting the first argument to our function
to be the size of the arrays, and you'll see that that is included as a
`mlir.ir_constant` parameter.

I'm not going to talk about the **JVP rule** here since it's quite problem
specific, but I've tried to comment the code reasonably thoroughly so check out
the code in `src/kepler_jax/kepler_jax.py` if you're interested, and open an
issue if anything isn't clear.

## Defining an XLA custom call on the GPU

The custom call on the GPU isn't terribly different from the CPU version above,
but the syntax is somewhat different and there's a heck of a lot more
boilerplate required. Since we need to compile and link CUDA code, there are
also a few more packaging steps, but we'll get to that in the next section. The
description in this section is a little all over the place, but the key files to
look at to get more info are (a) `lib/gpu_ops.cc` for the dispatch functions
called from Python, and (b) `lib/kernels.cc.cu` for the CUDA code implementing
the kernel.

The signature for the GPU custom call is:

```c++
// lib/kernels.cc.cu
template <typename T>
void gpu_kepler(
  cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
);
```

The first parameter is a CUDA stream, which I won't talk about at all because I
don't really know very much about GPU programming and we don't really need to
worry about it for now. Then you'll notice that the inputs and outputs are all
provided as a single `void**` buffer. These will be ordered such that our access
code from above is replaced by:

```c++
// lib/kernels.cc.cu
template <typename T>
void gpu_kepler(
  cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
) {
  const T *mean_anom = reinterpret_cast<const T *>(buffers[0]);
  const T *ecc = reinterpret_cast<const T *>(buffers[1]);
  T *sin_ecc_anom = reinterpret_cast<T *>(buffers[2]);
  T *cos_ecc_anom = reinterpret_cast<T *>(buffers[3]);
}
```

where you might notice that the `size` parameter is no longer one of the inputs.
Instead the array size is passed using the `opaque` parameter since its value is
required on the CPU and within the GPU kernel (see the [XLA custom
calls][xla-custom] documentation for more details). To use this `opaque`
parameter, we will define a type to hold `size`:

```c++
// lib/kernels.h
struct KeplerDescriptor {
  std::int64_t size;
};
```

and then the following boilerplate to serialize it:

```c++
// lib/kernel_helpers.h
#include <string>

// Note that bit_cast is only available in recent C++ standards so you might need
// to provide a shim like the one in lib/kernel_helpers.h
template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
  return std::string(bit_cast<const char*>(&descriptor), sizeof(T));
}

// lib/pybind11_kernel_helpers.h
#include <pybind11/pybind11.h>

template <typename T>
pybind11::bytes PackDescriptor(const T& descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}
```

This serialization procedure should then be exposed in the Python module using:

```c++
// lib/gpu_ops.cc
#include <pybind11/pybind11.h>

PYBIND11_MODULE(gpu_ops, m) {
  // ...
  m.def("build_kepler_descriptor",
        [](std::int64_t size) {
          return PackDescriptor(KeplerDescriptor{size});
        });
}
```

Then, to deserialize this descriptor, we can use the following procedure:

```c++
// lib/kernel_helpers.h
template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T*>(opaque);
}

// lib/kernels.cc.cu
template <typename T>
void gpu_kepler(
  cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
) {
  // ...
  const KeplerDescriptor &d = *UnpackDescriptor<KeplerDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;
}
```

Once we have these parameters, the full procedure for launching the CUDA kernel
is:

```c++
// lib/kernels.cc.cu
template <typename T>
void gpu_kepler(
  cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len
) {
  const T *mean_anom = reinterpret_cast<const T *>(buffers[0]);
  const T *ecc = reinterpret_cast<const T *>(buffers[1]);
  T *sin_ecc_anom = reinterpret_cast<T *>(buffers[2]);
  T *cos_ecc_anom = reinterpret_cast<T *>(buffers[3]);
  const KeplerDescriptor &d = *UnpackDescriptor<KeplerDescriptor>(opaque, opaque_len);
  const std::int64_t size = d.size;

  // Select block sizes, etc., no promises that these numbers are the right choices
  const int block_dim = 128;
  const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);

  // Launch the kernel
  kepler_kernel<T>
      <<<grid_dim, block_dim, 0, stream>>>(size, mean_anom, ecc, sin_ecc_anom, cos_ecc_anom);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}
```

Finally, the kernel itself is relatively simple:

```c++
// lib/kernels.cc.cu
template <typename T>
__global__ void kepler_kernel(
  std::int64_t size, const T *mean_anom, const T *ecc, T *sin_ecc_anom, T *cos_ecc_anom
) {
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
    compute_eccentric_anomaly<T>(mean_anom[idx], ecc[idx], sin_ecc_anom + idx, cos_ecc_anom + idx);
  }
}
```

## Building & packaging for the GPU

Since we're already using CMake to build our project, it's not too hard to add
support for CUDA. I've chosen to enable GPU builds whenever CMake can detect
CUDA support using `CheckLanguage` in `CMakelists.txt`:

```cmake
include(CheckLanguage)
check_language(CUDA)
```

Then, to expose this to JAX, we need to update the translation rule from above as follows:

```python
# src/kepler_jax/kepler_jax.py
import numpy as np
from jax.lib import xla_client
from kepler_jax import gpu_ops

for _name, _value in gpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

def _kepler_lowering_gpu(ctx, mean_anom, ecc):
    # Most of this function is the same as the CPU version above...

    # ...

    # The name of the op is now prefaced with 'gpu' (our choice, see lib/gpu_ops.cc,
    # not a requirement)
    if np_dtype == np.float32:
        op_name = "gpu_kepler_f32"
    elif np_dtype == np.float64:
        op_name = "gpu_kepler_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    # We need to serialize the array size using a descriptor
    opaque = gpu_ops.build_kepler_descriptor(size)

    # The syntax is *almost* the same as the CPU version, but we need to pass the
    # size using 'opaque' rather than as an input
    return custom_call(
        op_name,
        # Output types
        result_types=[dtype, dtype],
        # The inputs:
        operands=[mean_anom, ecc],
        # Layout specification:
        operand_layouts=[layout, layout],
        result_layouts=[layout, layout],
        # GPU-specific additional data for the kernel
        backend_config=opaque
    ).results

mlir.register_lowering(
        _kepler_prim,
        _kepler_lowering_gpu,
        platform="gpu")
```

Otherwise, everything else from our CPU implementation doesn't need to change.

## Testing

As usual, you should always test your code and this repo includes some unit
tests in the `tests` directory for inspiration. You can also see an example of
how to run these tests using the GitHub Actions CI service and the workflow in
`.github/workflows/tests.yml`. I don't know of any public CI servers that
provide GPU support, but I do include a test to confirm that the GPU ops can be
compiled. You can see the infrastructure for that test in the `.github/action`
directory.

## See this in action

To demo the use of this custom op, I put together a notebook, based on [an
example from the exoplanet docs][exoplanet-tutorial]. You can see this notebook
in the `demo.ipynb` file in the root of this repository or open it on Google
Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dfm/extending-jax/blob/main/demo.ipynb)

## References

[jax-primitives]: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html "How primitives work"
[xla-custom]: https://www.tensorflow.org/xla/custom_call "XLA custom calls"
[xla-adventures]: https://github.com/danieljtait/jax_xla_adventures "JAX XLA adventures"
[jaxlib]: https://github.com/google/jax/tree/master/jaxlib "jaxlib source code"
[keplers-equation]: https://en.wikipedia.org/wiki/Kepler%27s_equation "Kepler's equation"
[stan-cpp]: https://dfm.io/posts/stan-c++/ "Using external C++ functions with PyStan & radial velocity exoplanets"
[kepler-h]: https://github.com/dfm/extending-jax/blob/main/lib/kepler.h
[capsule]: https://docs.python.org/3/c-api/capsule.html "Capsules"
[jaxlib-lapack]: https://github.com/google/jax/blob/master/jaxlib/lapack.pyx "jax/lapack.pyx"
[scikit-build-core]: https://github.com/scikit-build/scikit-build-core "scikit-build-core"
[pybind11-cmake]: https://pybind11.readthedocs.io/en/stable/compiling.html#building-with-cmake "Building with CMake"
[exoplanet-tutorial]: https://docs.exoplanet.codes/en/stable/tutorials/intro-to-pymc3/#A-more-realistic-example:-radial-velocity-exoplanets "A more realistic example: radial velocity exoplanets"
