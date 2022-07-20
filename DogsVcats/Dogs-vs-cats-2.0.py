"""Used Full dataset and added augmentation and used dropout to the dataset"""
"""Full Dataset Downloaded from Kaggle"""

# todo When TRAINING THE MODEL PLUG IN THE LAPTOP "IT SIGNIFICANTLY INCREASES GPU PERFORMANCE"
# todo use transfer learning to create a more accurate version
import tensorflow as tf
from  tensorflow.keras.optimizers import RMSprop
from acc_plotter import plot_accuracy
import os
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt

def main_menu():
    print("PRESS -'Y'- TO TRAIN THE MODEL (WARNING IT CAN TAKE SEVERAL HOURS!!!- fir mat kehna bataya nahi tha)")
    print("-------------OR----------------")
    print("PRESS -'N'- JUST USE IT'S FUNCTIONALITY USING THE PRETRAINED MODEL")
    choice = input("Enter your choice : ")

    if choice.upper() == 'Y':
        model = model_create()
        model.summary()
        user_input = input("Do you want to load training data from previous Training(y/n): ")
        if user_input.lower() == 'y':
            path_to_weights = r'CheckPoints\Dogs-vs-cats-2.0\best_weights.hdf5'
            model.load_weights(path_to_weights)
        (train_dir,testing_dir) = directories()
        (train_generator, validation_generator) = image_data_generator(train_dir,testing_dir)
        callbacks = myCallback()
        cp_callbacks = cp_callback()
        history = train_model(model, train_generator, validation_generator, callbacks, cp_callbacks)
        plot_accuracy(history)
        check_user_image(model)

    elif choice.upper() =='N':
        model = model_create()
        model.load_weights(r'CheckPoints\Dogs-vs-cats-2.0\best_weights.hdf5')
        model.trainable= False
        (train_dir, testing_dir) = directories()
        (train_generator, validation_generator) = image_data_generator(train_dir, testing_dir)
        model.evaluate(validation_generator, verbose=1, steps=25 ) #steps = 25 for FULL DATASET
        check_user_image(model)
        # visualise_image(model)

    else:
        print("ERROR IN USER INPUT!!!!!")
        main_menu()


def directories():
    # Path of "Full" testing and training Directories
    train_dir = r'dogs_vs_cats_dataset_full\Validation_full_dataset\training'
    testing_dir = r'dogs_vs_cats_dataset_full\Validation_full_dataset\testing'
    # # Path of "Filtered" testing and training Directories
    # train_dir = r'dogs_vs_cats_dataset_full\train'
    # testing_dir = r'dogs_vs_cats_dataset_full\validation'

    return train_dir,testing_dir


def model_create():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(16, (3,3), input_shape=(150,150,3), activation='relu'),
            tf.keras.layers.BatchNormalization(trainable=True),
            tf.keras.layers.MaxPool2D(2,2),

            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(trainable=True),
            tf.keras.layers.MaxPool2D(2,2),

            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.BatchNormalization(trainable=True),
            tf.keras.layers.MaxPool2D(2,2),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(trainable=True),
            tf.keras.layers.MaxPool2D(2, 2),

            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ]
    )
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_model(model, train_generator, validation_generator, callbacks, cp_callbacks):
    history = model.fit(train_generator, epochs=20
                        , verbose=1,  validation_data=validation_generator, callbacks=[callbacks, cp_callbacks],
                        # steps_per_epoch=27 ,validation_steps=10)            #For filtered DATASET
                        steps_per_epoch=300, validation_steps=25)          # For FULL DATASET
    return history

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.9:
            print('\nValidation Accuracy reached 90% Stopping Training')
            self.model.stop_training = True


def cp_callback():
    checkpoint_path = r'CheckPoints\Dogs-vs-cats-2.0\best_weights.hdf5'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("Check point data will be stored in:", checkpoint_dir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     verbose=1, save_best_only=True,
                                                     save_weights_only=True)
    return cp_callback

def image_data_generator(train_dir,testing_dir):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale= 1/255,
        width_shift_range= 0.2,
        height_shift_range= 0.2,
        horizontal_flip= True,
        rotation_range= 45,
        zoom_range= 0.2,
        fill_mode= 'nearest'
        )

    validation_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=75,
        class_mode='binary'
        )

    validation_generator = validation_datagen.flow_from_directory(
        testing_dir,
        target_size=(150,150),
        batch_size=100,
        class_mode='binary'
        )

    return train_generator,validation_generator


def check_user_image(model):
    from tensorflow.keras.preprocessing import image
    import numpy as np

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
        x = x / 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        plt.imshow(images[0])
        plt.show()

        result = model.predict(images)
        print(result[0])
        if result[0] > 0.5:
            print('Image is Dog')
        elif result[0] < 0.5:
            print('Image is Cat')

        visualise_user_image(model, img)

        plt.imshow(images[0])
        plt.show()
        choice = input('Do you want to Test more Images??(y/n): ')
        choice = choice.lower()

def visualise_image(model):
    train_dir, test_dir = directories()
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)
    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]

    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

    # Let's prepare a random input image of a cat or dog from the training set.
    cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
    dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

    img_path = random.choice(cat_img_files + dog_img_files)
    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255.0

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):

        if len(feature_map.shape) == 4:

            # -------------------------------------------
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            # -------------------------------------------
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # -------------------------------------------------
            # Postprocess the feature to be visually palatable
            # -------------------------------------------------
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                # x -= x.mean()
                # x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = x  # Tile each filter into a horizontal grid

            # -----------------
            # Display the grid
            # -----------------

            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()

def visualise_user_image(model, img):
    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after
    # the first.
    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

    # Let's prepare a input image of a cat or dog from user input.
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255.0

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------
    count = 0
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):

        if len(feature_map.shape) == 4:

            # -------------------------------------------
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            # -------------------------------------------
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # -------------------------------------------------
            # Postprocess the feature to be visually palatable
            # -------------------------------------------------
            for i in range(n_features):
                x = feature_map[0, :, :, i]
                # x -= x.mean()
                # x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = x  # Tile each filter into a horizontal grid

            # -----------------
            # Display the grid
            # -----------------

            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()
            count +=1

        # Limit Plots
        if count == 6:
            break

main_menu()
