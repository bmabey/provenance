from provenance.hashing import hash
from conftest import artifact_record
import provenance.utils as u
import provenance.repos as r
import provenance.core as pc
import provenance as p
import conftest as c

import toolz as t
import pandas as pd
import numpy as np

import cloudpickle as pickle
import os
import random
import shutil
import tempfile
from copy import copy, deepcopy

import pytest
pytest.importorskip("torch")
import torch

class TwoLayerNet(torch.nn.Module):
    """
    This class is copied from PyTorch's documentation and is meant to be the
    simplest, non-trivial custom NN we can use for testing provenance.
    See [here](https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_module.html#sphx-glr-beginner-examples-nn-two-layer-net-module-py)
    """

    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them
        as member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must
        return a Tensor of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def random_data(N, D_in, D_out):
    """
    Generates random data for training/testing the PyTorch model.

    N is the data size
    D_in is the input dimension
    D_out is the output dimension
    """

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    return {
        'X_train': x,
        'Y_train': y,
        'X_test': x,
        'Y_test': y
    }


@p.provenance(returns_composite=True)
def fit_model(N=64, D_in=1000, D_out=10, H=100, epochs=500, seed=None):
    """
    An example workflow that provenance can handle from PyTorch. The model
    parameters, the data parameters, and the fit parameters are all passed
    into this function, and the output includes the PyTorch model and some
    metadata regarding its fit history (a list of losses after each epoch).
    """
    if seed is not None:
        torch.manual_seed(seed)

    data = random_data(N, D_in, D_out)
    x = data['X_train']
    y = data['Y_train']

    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to
    # model.parameters() in the SGD constructor will contain the learnable
    # parameters of the two nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    losses = []
    for t in range(epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        losses.append(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {'model': model, 'losses': losses}


def test_same_models_are_equal(dbdiskrepo):
    """
    Validates that two separately constructed models using the same parameters
    hash to the same artifact in provenance terms.
    """
    fit1 = fit_model()
    fit2 = fit_model()
    assert fit1.artifact.id == fit2.artifact.id
    assert fit1.artifact.value_id == fit2.artifact.value_id
    assert hash(fit1) == hash(fit2)


def test_copied_models_are_equal(dbdiskrepo):
    """
    Validates that a copied model (deep or shallow copied) hashes to the same
    artifact as the original in provenance terms.
    """
    original = fit_model()

    shallow = copy(original)
    assert original.artifact.id == shallow.artifact.id
    assert original.artifact.value_id == shallow.artifact.value_id
    assert hash(original) == hash(shallow)

    deep = deepcopy(original)
    assert original.artifact.id == deep.artifact.id
    assert original.artifact.value_id == deep.artifact.value_id
    assert hash(original) == hash(deep)


def test_reloading_from_disk_has_same_value_id(dbdiskrepo):
    """
    Validates that we can write and read a pytorch model as an artifact and that
    it is the same going in as coming out.
    """
    original = fit_model()
    loaded = p.load_proxy(original.artifact.id)

    assert loaded.artifact.value_id == p.hash(loaded.artifact.value)
    assert loaded.artifact.value_id == original.artifact.value_id
    assert loaded.artifact.id == original.artifact.id


def test_different_seeds_result_in_different_models(dbdiskrepo):
    """
    Validates that using different pytorch seeds to the fit model results in
    the same artifact.
    """
    fit1 = fit_model(seed=0)
    fit2 = fit_model(seed=1)

    assert p.hash(fit1) != p.hash(fit2)
    assert fit1.artifact.id != fit2.artifact.id
    assert fit1.artifact.value_id != fit2.artifact.value_id


def test_same_seeds_result_in_same_models(dbdiskrepo):
    """
    Validates that using the same pytorch seed to the fit model results in
    different artifacts.
    """
    fit1 = fit_model(seed=0)
    fit2 = fit_model(seed=0)

    assert p.hash(fit1) == p.hash(fit2)
    assert fit1.artifact.id == fit2.artifact.id
    assert fit1.artifact.value_id == fit2.artifact.value_id
