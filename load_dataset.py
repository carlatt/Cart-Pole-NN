from numpy import loadtxt
def load_data():
    inp = loadtxt("datasetModelInput.txt", comments="#", delimiter=" ", unpack=False)
    outp = loadtxt("datasetModelOutput.txt", comments="#", delimiter=" ", unpack=False)
    x_train = inp[:int(3*len(inp)/4)]
    y_train = outp[:int(3*len(inp)/4)]
    x_test = inp[int(3*len(inp)/4):]
    y_test = outp[int(3*len(inp)/4):]
    return  (x_train, y_train), (x_test, y_test)
lines = loadtxt("datasetModelInput.txt", comments="#", delimiter=" ", unpack=False)
print(lines[0])