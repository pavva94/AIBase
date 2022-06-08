from tensorflow.keras import layers, models


class CNN(models.Model):
  def __init__(self, class_len, input_shape):
    super(CNN, self).__init__()
    self.conv1 = layers.Conv2D(32, 3, activation='relu', input_shape=input_shape)
    self.conv2 = layers.Conv2D(64, 3, activation='relu')
    self.conv3 = layers.Conv2D(128, 3, activation='relu')
    self.flatten = layers.Flatten()
    self.d1 = layers.Dense(128, activation='relu')
    self.d2 = layers.Dense(class_len)
    self.pool1 = layers.MaxPooling2D((2, 2))
    self.pool2 = layers.MaxPooling2D((2, 2))

  def call(self, x, **kwargs):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))

