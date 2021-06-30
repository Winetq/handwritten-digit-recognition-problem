import numpy as np

from os import listdir
from os.path import isfile, join
from numpy import asarray
from PIL import Image


def load_digits():
    directories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    files = []
    for directory in directories:
        sub_dir = [f for f in listdir('digits/' + str(directory) + "/") if isfile(join('digits/' + str(directory) + "/", f))]
        files.append(sub_dir)

    images = []
    for i in directories:
        for file in files[i]:
            gray_image = np.zeros((28, 28), dtype=int)
            temp_image = Image.open('digits/' + str(i) + "/" + str(file))
            image_RGB = temp_image.convert('RGB')
            r, g, b = image_RGB.split()
            r = asarray(r)
            g = asarray(g)
            b = asarray(b)
            for j in range(28):
                for k in range(28):
                    pixel = r[j][k] * 0.299 + g[j][k] * 0.587 + b[j][k] * 0.114
                    negative = 255 - pixel
                    gray_image[j][k] = negative
            images.append(gray_image)

    return images, files, len(images)


def format_photos(images_as_matrices):
    images_as_vectors = np.zeros((len(images_as_matrices), 784), dtype=int)
    counter = 0
    for image in images_as_matrices:
        tmp_image = np.zeros(784, dtype=int)
        pixels_counter = 0
        for j in range(len(image)):
            for k in range(len(image)):
                tmp_image[pixels_counter] = image[j][k]
                pixels_counter += 1
        images_as_vectors[counter] = tmp_image
        counter += 1

    return images_as_vectors


def build_answers(digits, n):
    answers = np.zeros(n, dtype=int)
    counter = 0
    answer = 0

    for i in digits:
        for _ in i:
            answers[counter] = answer
            counter += 1
        answer += 1

    return answers
