import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from loading import load_digits, build_answers


def visualize_history(history: tf.keras.callbacks.History) -> None:
    """
    Visualize history of the training model.

    Parameters
    ----------
    history : tf.keras.callbacks.History
    """
    df_hist = pd.DataFrame(history.history)

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(9)
    fig.set_figwidth(16)

    axs[0].plot(df_hist["loss"], label="zbiór uczący")
    axs[0].plot(df_hist["val_loss"], label="zbiór walidacyjny")
    axs[0].set_title('Wartość funkcji kosztu podczas uczenia modelu')
    axs[0].set_xlabel('epoka')
    axs[0].set_ylabel('wartość')
    axs[0].legend()

    axs[1].plot(df_hist["accuracy"], label='zbiór uczący')
    axs[1].plot(df_hist["val_accuracy"], label='zbiór walidacyjny')
    axs[1].set_title('Skuteczności modelu podczas uczenia')
    axs[1].set_xlabel('epoka')
    axs[1].set_ylabel('wartość')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.15)

print(x_train.shape)
print(x_test.shape)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

for train in range(len(x_train)):
    for row in range(28):
        for col in range(28):
            if x_train[train][row][col] != 0:
                x_train[train][row][col] = 1

for test in range(len(x_test)):
    for row in range(28):
        for col in range(28):
            if x_test[test][row][col] != 0:
                x_test[test][row][col] = 1

model = tf.keras.models.Sequential()  # it creates an empty model object

model.add(tf.keras.layers.Flatten())  # it converts an N-dimentional layer to a 1D layer

# hidden layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# output layer - the number of neurons must be equal to the number of classes
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

visualize_history(history)

model.save('model.model')

print("Model saved\n")

# TESTING

model = tf.keras.models.load_model('model.model')

images, digits, n = load_digits()

images = tf.keras.utils.normalize(images, axis=1)

for i in range(n):
    for j in range(28):
        for k in range(28):
            if images[i][j][k] != 0:
                images[i][j][k] = 1

predictions = model.predict(images[:n])

y_prediction = []
error = 0
i = 0
for digit in range(len(digits)):
    for _ in range(len(digits[digit])):
        guess = (np.argmax(predictions[i]))  # max argument
        actual = digit
        print("Prediction: " + str(guess))
        print("Correct answer: " + str(actual))
        y_prediction.append(guess)
        if guess != actual:
            error += 1
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))  # cmap - colormap
        plt.show()
        i += 1

print("\nAccuracy = " + str(((len(predictions)-error)/len(predictions)) * 100) + "%")

answers = build_answers(digits, n)

# confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
confusion_matrix = confusion_matrix(y_true=answers, y_pred=y_prediction)

# heatmap
sns = sns.heatmap(confusion_matrix, annot=True, cmap='nipy_spectral_r', fmt='g')
sns.set_title("confusion matrix")
plt.show()
