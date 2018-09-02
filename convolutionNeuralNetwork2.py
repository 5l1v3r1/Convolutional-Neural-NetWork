# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:12:09 2018

@author: kjwil
"""

#Part 1 - Building Convolutional Neural Network
# import libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initialising the Convolutional Neural Network
classifier = Sequential()

# Step 1 - Convolution (creating Feature maps)
classifier.add(Conv2D(32,(3,3), input_shape = (64,64,3), activation = "relu"))

# Step 2 - Max Pooling reduced feature map, maintaining features
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding another convolutional layer to increase accuracy and decrease over fitting
classifier.add(Conv2D(32,(3,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding another convolutional layer to increase accuracy and decrease over fitting
classifier.add(Conv2D(32,(3,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening turn the feature map into one vector (column)
classifier.add(Flatten())
 
#Step 4 Full connection
 
classifier.add(Dense(units = 64, activation = "relu" ))
classifier.add(Dropout(0.5))
classifier.add(Dense(units =  1, activation = "sigmoid" ))

#compiling convolution neural network

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Part 2 - Fitting the CNN to the images. 
#import ImageDataGenerator, this functions gets all the images and changes them
# to help with overfitting issues, you can change target_size to larger area
# but this will need a gpu to gain results within a timely manner a cpu will
# take possibly days to process. If you do change target_size you will also need
# to change the input_shape to the same size.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
#############################################################################
#Only run if you want to train your own network
#############################################################################
print("Training the model.....")
classifier.fit_generator(
                            training_set,
                            steps_per_epoch=(8000/32),
                            epochs=90,
                            validation_data=test_set,
                            validation_steps=(2000/32))
#############################################################################
#############################################################################

# Heres what I prepared earlier load this before testing data, unless you used 
# the above code and created your own.
print("Loading model...")
from keras.models import load_model
classifier1 = load_model("model_1.h5")

#save model...really need this after 20/14hrs training!!
# Update to above comment, divided training set and test set to cut processing time drastically
# and still have a good enough result. Code now finishes in about 1 hour :)
print("Saving model....")
classifier.save("model_1.h5")

# predict against images not in test file or training file
import numpy as np
from keras.preprocessing import image 

# change file to what ever picture you want tested answer will always
# be a cat or dog...so if you try insert an image of a turtle do not be disapointed!
newTest = image.load_img("single_prediction/cat_or_dog_2.jpg", target_size =(64,64))
newTest = image.img_to_array(newTest)
newTest = np.expand_dims(newTest, axis = 0)
result = classifier.predict_classes(newTest)
training_set.class_indices
if result[0][0] == 1:
    prediction = "Dog"
else:
    prediction = "Cat"

#summary of the network
classifier.summary()
# shows all the weights its trained and used to make predictions
classifier.get_weights()
# reveals what optimizer was used
classifier.optimizer

classifier.predict_proba()

# Epoch 90/90
# loss: 0.1997 - acc: 0.9197 - val_loss: 0.4445 - val_acc: 0.8410
    
