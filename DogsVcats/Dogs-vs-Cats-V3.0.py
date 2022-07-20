"""Used Full dataset and added Transfer learning to improve upon V2"""
"""Full Dataset Downloaded from Kaggle"""
"""Transfer learning: https://www.tensorflow.org/tutorials/images/transfer_learning"""

# todo When TRAINING THE MODEL PLUG IN THE LAPTOP "IT SIGNIFICANTLY INCREASES GPU PERFORMANCE"

import tensorflow as tf
import numpy as np
from acc_plotter import plot_accuracy
from tensorflow.keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt

# Path of "Full" testing and training Directories
train_dir = r'dogs_vs_cats_dataset_full\Validation_full_dataset\training'
testing_dir = r'dogs_vs_cats_dataset_full\Validation_full_dataset\testing'

# # Path of "Filtered" testing and training Directories
# train_dir = r'dogs_vs_cats_dataset_full\train'
# testing_dir = r'dogs_vs_cats_dataset_full\validation'

load_weights_file = r'Inception_V3_weights\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Loading INCEPTIONV3's conv layers or the bottom layers except for the top layers to raise our programs accuracy
pre_trained_model = InceptionV3(input_shape=(150,150,3), include_top=False, weights=None)
pre_trained_model.load_weights(load_weights_file)

for layer in pre_trained_model.layers:          # Use pre_trained_mode.trainable=False
    layer.trainable='False'

last_layer = pre_trained_model.get_layer('mixed7')
print('Last Layer Output Shape : ', last_layer.output_shape)
last_output = last_layer.output


# Declaring Our Own Layers to work on top of inceptionV3
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# todo you can use this to convert the program into a more familiar category
"""model = tf.keras.Sequential([
  base_model,
  user_defined_layers
])"""

model = tf.keras.Model(pre_trained_model.input,x)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), metrics=['accuracy'])

#Callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.999:
            print('\nValidation Accuracy reached 90% Stopping Training')
            self.model.stop_training = True
callbacks = myCallback()

# Creating ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(rescale=1/255,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      zoom_range=0.2,
                                      rotation_range=45,
                                      shear_range=0.2,
                                      fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1/255)

training_generator = training_datagen.flow_from_directory(train_dir,
                                                          batch_size=75,
                                                          class_mode='binary',
                                                          shuffle=True,
                                                          target_size=(150,150))

val_generator = val_datagen.flow_from_directory(testing_dir,
                                                batch_size=100,
                                                target_size=(150,150),
                                                class_mode='binary',
                                                )

history = model.fit(training_generator,
                    epochs=25,
                    verbose=1,validation_data=val_generator, callbacks=[callbacks],
                    steps_per_epoch=27 ,validation_steps=10)            #For filtered DATASET
                    # steps_per_epoch=300, validation_steps=25)          # For FULL DATASET

plot_accuracy(history)

from tensorflow.keras.preprocessing import image

choice = input('Do you want to Test on a Images??(y/n): ')
choice = choice.lower()
while choice == 'y':
    try:
        path = input('Input the Path to the image  file :')
        img = image.load_img(path, target_size=(150, 150))
    except:
        print("error in user input!")
        path = input('Input the Path to the image  file :')
        img = image.load_img(path, target_size=(150, 150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    result = model.predict(images)
    print(result[0])
    if result[0] > 0.5:
        print('Image is Dog')
    elif result[0] < 0.5:
        print('Image is Cat')
    plt.imshow(image)
    plt.show()
    choice = input('Do you want to Test more Images??(y/n): ')
    choice = choice.lower()