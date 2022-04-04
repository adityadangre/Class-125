import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

x, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(y).value_counts())

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 2500, train_size = 7500)
x_train_scalled = x_train / 255
x_test_scalled = x_test / 255

classifier = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scalled, y_train)

def prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_scaled = np.clip(image_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_scaled = np.asarray(image_bw_resized_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_scaled).reshape(1,784)
    test_predict = classifier.predict(test_sample)
    return test_predict[0]