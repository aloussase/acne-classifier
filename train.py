import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths to your dataset
train_data_dir = './datasets/train'
val_data_dir = './datasets/validation'

# Set parameters
img_width, img_height = 32, 32
batch_size = 32
epochs = 30

# Data augmentation for training dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

# Rescale validation dataset
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and augment training dataset
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Load validation dataset
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Build the neural network model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
)


def test_model(model):
    test_data_dir = './datasets/test'
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )
    test_score = model.evaluate(test_generator, verbose=0)
    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])


def plot_accuracy(history):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(history.history['accuracy'], ax=ax, label='Training Accuracy')
    sns.lineplot(history.history['val_accuracy'], ax=ax, label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    fig.savefig('accuracy.png')


def plot_loss(history):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.lineplot(history.history['loss'], ax=ax, label='Training loss')
    sns.lineplot(history.history['val_loss'], ax=ax, label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig.savefig('loss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Acne Classifier")

    parser.add_argument('-s', '--save', action='store_true', help='Save the model after training')
    parser.add_argument('-t', '--test', action='store_true', help='Test the model after training')

    args = parser.parse_args()

    plot_accuracy(history)
    plot_loss(history)

    if args.test:
        test_model(model)

    if args.save:
        model.save('acne_model.h5')
