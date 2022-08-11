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
sys.path.append('../../samples/NumpyroExamples/CoinToss')
sys.path.append('../')

# Import the model and the converter
from CoinToss import coin_toss
from converters import Converter
from potential import statisticalModelPotential, statisticalModelGradient
# Load the observed outcomes and the reference biases
data = json.load(open("../../samples/NumpyroExamples/CoinToss/CoinToss.data.json"))

# store the user-provided 'true' values of the biases
p1_reference = float(data["p1"])
p2_reference = float(data["p2"])

referenceParameterDictionary = {'p1': p1_reference,
                                'p2': p2_reference
                                }
modelDataDictionary = { 'c1': np.array(data["c1"]),
                        'c2': np.array(data["c2"])}
parameterVector = np.array([ p1_reference, p2_reference])

# Prepare the kernel and run
model = coin_toss

# Create the converter object

typeConverter = Converter(model, (), modelDataDictionary)

# Simple test of proper functioning of the converter

convertedDictionary = typeConverter.toDict(parameterVector)

for key in convertedDictionary.keys():
    assert convertedDictionary[key] == referenceParameterDictionary[key], "Dictionary elements not identical"

print("Dictionary conversion successful!")
print("Reference: ", referenceParameterDictionary)
print("Obtained: ", convertedDictionary)

convertedVector = typeConverter.toArray(referenceParameterDictionary)

assert convertedVector.size == parameterVector.size, "Vector sizes do not match for converted vector!"

for i in range(convertedVector.size):
    assert convertedVector[i] == parameterVector[i], "Vector elements differ"


print("Array conversion succesful!")
print("Reference: ", parameterVector)
print("Obtained: ", convertedVector)

# testing the potential functions

potentialValue = statisticalModelPotential(model, parameterVector, typeConverter, (), modelDataDictionary)

print("Computed potential value :", potentialValue)

grad = statisticalModelGradient(model, parameterVector, typeConverter, (), modelDataDictionary)

print("Computed gradient :", grad)
