from keras.applications import resnet50
from keras import Sequential
from keras.layers import Dense
from PIL import Image
import glob
import numpy as np
import pyspark

def import_data(filepath, num_files=None):
    image_list = []
    age_list = []
    i=0
    for filename in glob.glob(filepath+'/*.jpg'):
        age = filename.split('_')[0].split('/')[-1]
        age_list.append(age)
        im = Image.open(filename)
        pixels = list(im.getdata())
        width, height = im.size
        pixels = np.array(pixels).reshape((width, height, 3))
        image_list.append(pixels)
        if i == num_files:
            break
        i+=1
    return image_list,age_list

filepath = r"./UTKFace"
images,ages = import_data(filepath,100)

sc = pyspark.SparkContext("local")