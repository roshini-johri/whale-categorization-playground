
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

from tqdm import tqdm


train_images = glob("/home/rjohri/whale-categorization-playground/data/train/*.jpg")
test_images = glob("/home/rjohri/whale-categorization-playground/data/test/*.jpg")
df = pd.read_csv("~/whale-categorization-playground/data/train.csv")

df["Image"] = df["Image"].map( lambda x : "data/train/"+x)
ImageToLabelDict = dict( zip( df["Image"], df["Id"]))

SIZE = 100
print(len(train_images))
def ImportImage( filename):
    img = Image.open(filename).resize( (SIZE,SIZE))
    img = np.array(img)
    if img.ndim == 2: #imported BW picture and converting to "dumb RGB"
        img = np.tile( img, (3,1,1)).transpose((1,2,0))
    return img
x_train = np.array([ImportImage(img) for img in train_images],dtype=np.uint8)


print( "%d training images" %x_train.shape[0])

print( "Nbr of samples/class\tNbr of classes")
for index, val in df["Id"].value_counts().value_counts().sort_index().iteritems():
    print( "%d\t\t\t%d" %(index,val))

