import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from loading import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

mnist_train = pd.read_csv('mnist_train.csv')

data = mnist_train.values

x = data[:, 1:]
y = data[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

classifier = KNeighborsClassifier(n_neighbors=15)

classifier.fit(x_train, y_train)

# TESTING

images, digits, n = load_digits()

# format photos to the proper format for the classifier
x_test = format_photos(images)

y_test = build_answers(digits, n)

y_prediction = classifier.predict(x_test)

i = 0
for digit in range(len(digits)):
    for _ in range(len(digits[digit])):
        guess = y_prediction[i]
        actual = digit
        print("Prediction: " + str(guess))
        print("Correct answer: " + str(actual))
        plt.imshow(images[i], cmap=plt.get_cmap('gray'))  # cmap - colormap
        plt.show()
        i += 1

# check accuracy
accuracy = metrics.accuracy_score(y_test, y_prediction)
print("\nAccuracy: ", accuracy)

# confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
confusion_matrix = confusion_matrix(y_true=y_test, y_pred=y_prediction)

# heatmap
sns = sns.heatmap(confusion_matrix, annot=True, cmap='nipy_spectral_r', fmt='g')
sns.set_title("confusion matrix")
plt.show()
