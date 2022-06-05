                                            ###########  Image-classification-with-CNN  ##########
                                        
# This work consists in creating a CNN model for the classification of two types of images, images from the movie "Toy Story", 
# and other images from the movie "Ernest & Celestine". This CNN model will be trained on 500 images (size 920x540) for both types of images.

## Importing libraries 

from tensorflow import keras
from tensorflow.keras import layers 
from keras.models import load_model 
from sklearn.model_selection import train_test_split 
from os import listdir 
from os.path import isfile, join
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

#.............................................................................................................
#.............................................................................................................

## Import of all the images (500 images) of type "Ernest & Celestine" in our file 
mypath='ernest_celestine_01'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
    images[n] = np.asarray(images[n]).astype("float32") 
    images[n] = images[n]/255 # pour la normalisation
    images[n] = cv2.resize(images[n], (0, 0), fx=0.4, fy=0.4, interpolation = cv2.INTER_NEAREST) # reduce the size of the images to facilitate and accelerate the processing of our model

## Import of all images (500 images) of type "Toy Story" in our file 
mypath2='toy_story_3_01'
onlyfiles2 = [ f for f in listdir(mypath2) if isfile(join(mypath2,f)) ]
images2 = np.empty(len(onlyfiles2), dtype=object)
for n in range(0, len(onlyfiles2)):
    images2[n] = cv2.imread( join(mypath2,onlyfiles2[n]))
    images2[n] = np.asarray(images2[n]).astype("float32")
    images2[n] = images2[n]/255 # pour la normalisation
    images2[n] = cv2.resize(images2[n], (0, 0), fx=0.4, fy=0.4, interpolation = cv2.INTER_NEAREST) # reduce the size of the images to facilitate and accelerate the processing of our model
    
#.............................................................................................................
#.............................................................................................................

## Preparation of training and validation data

X_film1 = images # "Ernest & Celestine" images
Y_film1 = np.ones(500) # label "Ernest & Celestine" images with 1

X_film2 = images2 # "Toy Story" images
Y_film2 = np.zeros(500) # label "Toy Story" images with 0

# Preparation of a list of images of dimension 4 of the 2 types of images for our model
XX1 = X_film1[0]
for i in range(1,len(X_film1)):
    XX1 = np.concatenate((XX1, X_film1[i]),axis=0)
X_train1 = XX1.reshape((500, 216, 368, 3)) 
    
XX2 = X_film2[0]
for i in range(1,len(X_film2)):
    XX2 = np.concatenate((XX2, X_film2[i]),axis=0)
X_train2 = XX2.reshape((500, 216, 368, 3))

# Assemble all images of both types in X_train and their labels in Y_train
X_train = np.concatenate((X_train1, X_train2),axis=0)
Y_train = np.concatenate((Y_film1, Y_film2),axis=0)

# Split the images between training (50%) and validation (50%) data and mix them
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X_train, Y_train, test_size=0.5, random_state=42)

#.............................................................................................................
#.............................................................................................................

## Creation of our CNN model, which can be summarized as follows:
#  Input → Convolution → ReLU → MaxPooling → Convolution → ReLU → MaxPooling → FC → ReLU → FC → Sigmoid → Output 

model = keras.models.Sequential()
model.add( keras.layers.Input((216, 368, 3)) )
model.add( keras.layers.Conv2D(16, (3,3),  activation='relu') )
model.add( keras.layers.MaxPooling2D((2,2)))
model.add( keras.layers.Conv2D(32, (3,3), activation='relu') )
model.add( keras.layers.MaxPooling2D((2,2)))
model.add( keras.layers.Flatten())
model.add( keras.layers.Dense(50, activation='relu'))
model.add( keras.layers.Dense(1, activation = 'sigmoid'))

model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#.............................................................................................................
#.............................................................................................................

## Training of the model on all data (Xtrain, Ytrain, Xvalid, Yvalid)
history = model.fit(Xtrain,
                    Ytrain,
                    epochs=20,
                    batch_size=32,
                    validation_data=(Xvalid, Yvalid))

#.............................................................................................................
#.............................................................................................................

## Display the results development for accuracy and loss
def Training_evolution(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train data', 'test data'], loc='upper left')
    plt.show()
    
    # summary of history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
Training_evolution(history)

#.............................................................................................................
#.............................................................................................................

## Save and import of the model
model.save("CNN_model.h5")
model_AE = load_model("CNN_model.h5")

#.............................................................................................................
#.............................................................................................................

# Test the model with images of both types of movies, and that are out of training and validation data
image = cv2.imread('toy-story-3-00050.png')
plt.imshow(image)
image = np.array(image).astype('float32')/255
image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4, interpolation = cv2.INTER_NEAREST)
image = np.expand_dims(image, axis=0)
pred = model1.predict(image)
if pred <0.5:
    prediction = 0
else:
    prediction = 1

classe_name = ['toy-story', 'ernest-celestine']
plt.title(classe_name[prediction])

#.............................................................................................................
#.............................................................................................................

## Result: the model has an accuracy of 97.4% on the test images after 20 iterations.