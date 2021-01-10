# -*- coding: utf-8 -*-

import numpy as np
import pytest

import jax
from jax.config import config
from jax.test_util import check_grads

from kepler_jax import kepler


config.update("jax_enable_x64", True)


@pytest.fixture(params=[np.float32, np.float64])
def kepler_data(request):
    # Note about precision: the precision of the mod function in float32 is not
    # great so we're only going to test values in the range ~0-2*pi. In real
    # world applications, the mod should be done in float64 even if the solve
    # is done in float32
    ecc = np.linspace(0, 0.9, 55)
    true_ecc_anom = np.linspace(-np.pi, np.pi, 101)
    mean_anom = true_ecc_anom - ecc[:, None] * np.sin(true_ecc_anom)
    dtype = request.param
    return (
        mean_anom.astype(dtype),
        ecc.astype(dtype),
        (true_ecc_anom + np.zeros_like(mean_anom)).astype(dtype),
    )


def check_kepler(sin_ecc_anom, cos_ecc_anom, true_ecc_anom):
    assert np.all(np.isfinite(sin_ecc_anom))
    np.testing.assert_allclose(sin_ecc_anom, np.sin(true_ecc_anom), atol=1e-5)
    assert np.all(np.isfinite(cos_ecc_anom))
    np.testing.assert_allclose(cos_ecc_anom, np.cos(true_ecc_anom), atol=1e-5)


def test_kepler(kepler_data):
    mean_anom, ecc, true_ecc_anom = kepler_data
    sin_ecc_anom, cos_ecc_anom = kepler(
        mean_anom, ecc[:, None] + np.zeros_like(mean_anom)
    )
    check_kepler(sin_ecc_anom, cos_ecc_anom, true_ecc_anom)


def test_kepler_broadcast(kepler_data):
    mean_anom, ecc, true_ecc_anom = kepler_data
    sin_ecc_anom, cos_ecc_anom = kepler(mean_anom, ecc[:, None])
    check_kepler(sin_ecc_anom, cos_ecc_anom, true_ecc_anom)


def test_kepler_jit(kepler_data):
    mean_anom, ecc, true_ecc_anom = kepler_data
    sin_ecc_anom, cos_ecc_anom = jax.jit(kepler)(mean_anom, ecc[:, None])
    check_kepler(sin_ecc_anom, cos_ecc_anom, true_ecc_anom)


def test_kepler_vmap(kepler_data):
    mean_anom, ecc, true_ecc_anom = kepler_data
    sin_ecc_anom, cos_ecc_anom = jax.vmap(kepler)(mean_anom, ecc)
    check_kepler(sin_ecc_anom, cos_ecc_anom, true_ecc_anom)


def test_kepler_grad(kepler_data):
    mean_anom, ecc, true_ecc_anom = kepler_data
    if mean_anom.dtype != np.float64:
        pytest.skip("Gradients only stable in double precision")

    m = ecc > 0.01
    check_grads(
        lambda *args: kepler(*args)[0],
        [mean_anom[m], ecc[m][:, None]],
        2,
        eps=1e-6,
    )
    check_grads(
        lambda *args: kepler(*args)[1],
        [mean_anom[m], ecc[m][:, None]],
        2,
        eps=1e-6,
    )
