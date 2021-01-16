import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, ZeroPadding2D, Dropout, Softmax

class AlexNet(Model):
    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        # 特征提取
        self.features = Sequential([
            ZeroPadding2D(((1, 2), (1, 2))),                                 # input(None, 224, 224, 3), output(None, 227, 227, 3)
            Conv2D(48, kernel_size=11, strides=4, activation="relu"),        # output(None, 55, 55, 48)
            MaxPool2D(pool_size=3, strides=2),                               # output(None, 27, 27, 48)
            Conv2D(128, kernel_size=5, padding="same", activation="relu"),   # output(None, 27, 27, 128)
            MaxPool2D(pool_size=3, strides=2),                               # output(None, 13, 13, 128)
            Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
            Conv2D(128, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 128)
            MaxPool2D(pool_size=3, strides=2)])                            # output(None, 6, 6, 128)

        # 展平
        self.flatten = Flatten()

        # 分类
        self.classifier = Sequential([
            Dropout(0.2),
            Dense(1024, activation="relu"),                           # output(None, 2048)
            Dropout(0.2),                                             # 减少过拟合
            Dense(128, activation="relu"),                            # output(None, 2048)
            Dense(num_classes, activation="softmax")                  # output(None, 10)
        ])

    def call(self, inputs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    input1 = tf.random.normal([32, 224, 224, 3])
    model = AlexNet(10)
    output = model(input1)
    print(output)