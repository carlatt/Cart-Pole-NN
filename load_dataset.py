import numpy
from numpy import loadtxt

lines = loadtxt("datasetModelInput.txt", comments="#", delimiter=" ", unpack=False)
print(lines[0])