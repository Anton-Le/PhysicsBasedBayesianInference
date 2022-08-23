import sys
import numpyro
import jax
import numpy as np
import json


numpyro.set_platform("cpu")


# Print run-time configuration infromation
print(f"jax version: {jax.__version__}")
print(f"numpyro version: {numpyro.__version__}")
print(f"jax target backend: {jax.config.FLAGS.jax_backend_target}")
print(f"jax target device: {jax.lib.xla_bridge.get_backend().platform}")
cpus = jax.devices("cpu")
print("Available CPUs:")
print(cpus)


# prepend the path to the model
sys.path.append("../../samples/NumpyroExamples/CoinToss")
sys.path.append("../")

# Import the model and the converter
from CoinToss import coin_toss
from converters import Converter
from potential import (
    statisticalModelPotential,
    statisticalModelGradient,
    statisticalModel,
)

# Load the observed outcomes and the reference biases
data = json.load(
    open("../../samples/NumpyroExamples/CoinToss/CoinToss.data.json")
)

# store the user-provided 'true' values of the biases
p1_reference = float(data["p1"])
p2_reference = float(data["p2"])

referenceParameterDictionary = {"p1": p1_reference, "p2": p2_reference}
modelDataDictionary = {"c1": np.array(data["c1"]), "c2": np.array(data["c2"])}
parameterVector = np.array([p1_reference, p2_reference])

# Prepare the kernel and run
model = coin_toss

# Create the converter object

typeConverter = Converter(model, (), modelDataDictionary)


def test_toDict():
    # Simple test of proper functioning of the converter
    convertedDictionary = typeConverter.toDict(parameterVector)

    for key in convertedDictionary.keys():
        assert (
            convertedDictionary[key] == referenceParameterDictionary[key]
        ), "Dictionary elements not identical"

    print("Dictionary conversion successful!")
    print("Reference: ", referenceParameterDictionary)
    print("Obtained: ", convertedDictionary)


def test_toArray():
    convertedVector = typeConverter.toArray(referenceParameterDictionary)
    assert (
        convertedVector.size == parameterVector.size
    ), "Vector sizes do not match for converted vector!"

    for i in range(convertedVector.size):
        assert (
            convertedVector[i] == parameterVector[i]
        ), "Vector elements differ"

    print("Array conversion succesful!")
    print("Reference: ", parameterVector)
    print("Obtained: ", convertedVector)


def test_modelGradient():
    """
    This function tests the computation of the gradient of
    the Coin Toss model at the true parameter values.
    Hence the expected result should be the 0-vector.
    """
    grad = statisticalModelGradient(
        model, parameterVector, typeConverter, (), modelDataDictionary
    )
    for i in range(grad.size):
        assert grad[i] == 0, "Gradient values erroneous!"
    # print("Computed gradient :", grad)


# Run the tests

test_toDict()
test_toArray()

# testing the potential functions

potentialValue = statisticalModelPotential(
    model, parameterVector, typeConverter, (), modelDataDictionary
)

print("Computed potential value :", potentialValue)

# perform the same test with the statisticalModel class

statModel = statisticalModel(model, (), modelDataDictionary)

potentialValue = statModel.potential(parameterVector)

print("Potential value computed with the model class: ", potentialValue)

test_modelGradient()

grad = statModel.grad(parameterVector)

print("Gradient computed with the model class: ", grad)
