import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from loading import *

# mnist_train - 60000 rows x 785 columns (785 columns - the first value is the label (a number from 0 to 9) and the
# remaining 784 values are the pixel values (a number from 0 to 255))
mnist_train = pd.read_csv('mnist_train.csv')

# mnist_test - 10000 rows x 785 columns
mnist_test = pd.read_csv('mnist_test.csv')

x_data_train = mnist_train.iloc[:, 1:]
y_data_train = mnist_train.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(x_data_train, y_data_train, test_size=0.15)

classifier = svm.SVC(C=0.95)

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
print("\nAccuracy = {}".format(accuracy))

# confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
confusion_matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_prediction)

# heatmap
sns = sns.heatmap(confusion_matrix, annot=True, cmap='nipy_spectral_r', fmt='g')
sns.set_title("confusion matrix")
plt.show()
