from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.activations import gelu
from tensorflow.keras import  Model

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='gelu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='gelu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)      # input[batch, 28, 28, 1]  output[batch, 26, 26, 1]
        x = self.flatten(x)    # output[batch, 26*26]
        x = self.d1(x)         # output[batch, 128]
        return self.d2(x)      # output[batch, 10]


