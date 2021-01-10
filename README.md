# Extending JAX with custom C++ and CUDA code

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

**A warning**: I'm writing this in January 2021 and much of what I'm talking
about is based on essentially undocumented APIs that are likely to change.
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

## Defining an XLA custom call on the CPU

## Building & packaging for the CPU

## Defining an XLA custom call on the GPU

## Exposing this op as a JAX primitive

## Testing

## References

[jax-primitives]: https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html "How primitives work"
[xla-custom]: https://www.tensorflow.org/xla/custom_call "XLA custom calls"
[xla-adventures]: https://github.com/danieljtait/jax_xla_adventures "JAX XLA adventures"
[jaxlib]: https://github.com/google/jax/tree/master/jaxlib "jaxlib source code"
[keplers-equation]: https://en.wikipedia.org/wiki/Kepler%27s_equation "Kepler's equation"
[stan-cpp]: https://dfm.io/posts/stan-c++/ "Using external C++ functions with PyStan & radial velocity exoplanets"
[kepler-h]: https://github.com/dfm/extending-jax/blob/main/src/kepler_jax/src/kepler.h
