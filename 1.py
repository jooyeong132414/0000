import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2  # 각도
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_number
    return X, y

X, y = spiral_data(samples=100, classes=3)

layer1 = Layer_Dense(2, 3)
layer1.forward(X)

layer2 = Layer_Dense(3, 5)
layer2.forward(layer1.output)

print("첫 번째 레이어 출력값:\n", layer1.output)
print("두 번째 레이어 출력값:\n", layer2.output)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method="xavier"):
        if initialize_method == "xavier":
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif initialize_method == "he":
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        elif initialize_method == "gaussian":
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(2, 3, initialize_method="he")
layer1.forward(X)
print("He 초기화 가중치로 계산된 첫 번째 레이어 출력값:\n", layer1.output)

layer1 = Layer_Dense(2, 3)
layer1.forward(X)

layer1.output = np.maximum(0, layer1.output)
print("ReLU 적용 후 출력값:\n", layer1.output)