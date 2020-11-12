# Based on: https://www.tensorflow.org/tutorials/images/segmentation

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from UNet import UNet

import matplotlib.pyplot as plt

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask


@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


dataset_train = tfds.load("coco", split='train[:20]')


TRAIN_LENGTH = len(dataset['train'])
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

model = UNet
tf.keras.utils.plot_model(model, show_shapes=True)

# early_stop = EarlyStopping(patience=10, verbose=1)
# checkpoint = ModelCheckpoint(os.path.join(model_save_path, "keras_unet_model.h5"),
#                              verbose=1, save_best_only=True)
# history = model.fit_generator(train_gen, steps_per_epoch=train_steps,
#                               epochs=50,
#                               validation_data=val_gen,
#                               validation_steps=val_steps,
#                               verbose=1,
#                               max_queue_size=4,
#                               callbacks=[early_stop, checkpoint])
#
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# miou = history.history["mean_iou"]
# val_miou = history.history["val_mean_iou"]
# epochs = range(1, len(loss) + 1)