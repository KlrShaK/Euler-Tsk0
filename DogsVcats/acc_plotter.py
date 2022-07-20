def plot_accuracy(history):
    # todo Accesing Training Data :
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc , 'b', label='Training Acc')
    plt.plot(epochs, val_acc, 'r', label='Validation_accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Training and validation Accuracy')
    plt.legend()
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and validation Loss')
    plt.legend()
    plt.show()  # Todo To Show the Graphs