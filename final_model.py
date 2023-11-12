import time as Time

from pathlib import Path
from matplotlib import pyplot as plt
from quickdraw import QuickDrawDataGroup, QuickDrawData

from tensorflow.keras.preprocessing import image_dataset_from_directory  # remember to install tf-nightly to fix this

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

from keras.applications import MobileNetV2

import tensorflow as tf
import keras
from keras import layers

image_size = (64, 64)  # this is the image resolution for the data

categories = ['airplane', 'apple', 'axe', 'basketball', 'bicycle', 'broccoli', 'bucket', 'butterfly',
              'crab', "diamond", 'fence', 'fish', "guitar", 'hammer', 'headphones', 'helicopter',
              'hot air balloon', 'ice cream', 'light bulb', 'lollipop', 'palm tree', 'parachute',
              'rainbow', 'sailboat', 'shoe', 'smiley face', 'star', "tennis racquet", 'traffic light',
              'wristwatch']  # categories for images in training data


#  loading the data
def generate_class_images(name, max_drawings, recognized):
    # this function creates a subdirectory named <name> in dataset_30 which has <max_drawings> images of class <name>.
    # the function only loads images that Google's network correctly identified

    directory = Path("dataset_30/" + name)  # specifying the path of the dataset
    directory_test = Path("dataset_30_test/" + name)  # specifying the path of the dataset

    if not directory.exists():
        directory.mkdir(parents=True)  # make a subdirectory for the category in dataset_30
        directory_test.mkdir(parents=True)  # make a subdirectory for the category in dataset_30_test

    images = QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)  # load the images using the
                                                                                         # quick draw library
    i = 1
    for img in images.drawings:
        if i <= 7000:
            filename = directory.as_posix() + "/" + str(img.key_id) + ".png"  # name the image
        else:
            filename = directory_test.as_posix() + "/" + str(img.key_id) + ".png"  # name the image

        img.get_image(stroke_width=3).resize(image_size).save(filename)  # make the image 64 by 64
        i = i + 1


c = True  # deciding whether to load the images or not

# generating 8400 images for each category in categories and putting 7000 of them in subdirectories in dataset_30
# and putting the other 1400 in subdirectories in dataset_30_test
if c:
    for label in categories:
        generate_class_images(label, max_drawings=8400, recognized=True)

print("\n\nFinished loading images for dataset_30 and dataset_30_test-------------------------------------------------")
# -------------------------------------------------------------------------------
load_dataset = True  # deciding whether to create the files used for the actual training or not

if load_dataset:
    # preparing the data
    batch_size = 128  # because we load the images using image_dataset_from_directory() they are
    # already saved in a format ready for training a neural network, so we need to specify the batch size

    # this function will return an image dataset where each image is paired with a label corresponding
    # to the class it belongs to (i.e. for classes a and b pictures from class a will have a label 0
    # and pictures from class b will have label 1)
    test_ds = image_dataset_from_directory(  # test dataset
        "dataset_30_test",
        seed=123,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    train_ds, val_ds = image_dataset_from_directory(  # training and validation dataset
        "dataset_30",
        validation_split=0.2,
        subset="both",
        seed=123,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    test_ds.save('test_dataset')
    train_ds.save('training_dataset')
    val_ds.save('validation_dataset')

test_ds = tf.data.Dataset.load('test_dataset')  # the test dataset
train_ds = tf.data.Dataset.load('training_dataset')  # the training dataset
val_ds = tf.data.Dataset.load('validation_dataset')  # the validation dataset
# --------------------------------------------------------------------------------------
# building, compling the training the model

input_shape = (image_size[0], image_size[1], 3)  # choosing the input shape of the model

n_classes = len(categories)  # number of categories in training set data

# creating the model we will train
model = Sequential([
    Rescaling(1. / 255, input_shape=input_shape),
    BatchNormalization(),

    Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
    Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(400, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(200, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(100, activation='relu'),
    Dropout(0.2),

    Dense(n_classes, activation='softmax')
])

# Train the model
opt = tf.optimizers.Adam()  # choosing the Adam optimizer


#  using the top k categorical accuracy which means it measures the number
#  of times the correct category of the image was in the 3 categories which
#  received the highest probability by the network and divides it by the
#  overall predictions the network made
top_k_categorical_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
    k=3, name='top_k_categorical_accuracy', dtype=None
)


model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',  # using the categorical_crossentropy
    # loss function because we are trying to differentiate between a number
    # of categories
    metrics=["accuracy", top_k_categorical_accuracy]
)


model.summary()  # printing a summary of the model's architecture
epochs = 4  # number of epochs the network will run for

tic = Time.time()  # starting to measure the time the network took to run

history = model.fit(  # training the model and saving the results to history
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
)

toc = Time.time()  # finishing to measure the time the network took to run

# Calculate training time and format as min:sec
minutes = format((toc - tic) // 60, '.0f')
sec = format(100 * ((toc - tic) % 60) / 60, '.0f')
print(f"Total training time (min:sec): {minutes}:{sec}")

results = model.evaluate(test_ds, return_dict=True)
print(results)

# -----------------------------------------------------------------------

# printing and graphing the results

print("Train accuracy: " + str(max(history.history['accuracy'])))
print("val_accuracy: " + str(max(history.history['val_accuracy'])))
print("test_accuracy: " + str(results['accuracy']))

plt.title('learning curves')
plt.xlabel('epoch')
plt.ylabel('Loss (cross entropy)')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

plt.title('learning curves')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='validation_acc')
plt.legend()
plt.show()


# ------------------------------------------------------------------------------------------------

# saving the model
name = "final_model"
model.save("models/" + name + ".h5")
print("Saved model to disk")
