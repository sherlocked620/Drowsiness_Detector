# -*- coding: utf-8 -*-
"""Drowsiness_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1r5qfAzNhyRzr5sj93aNU23qNJBjElGsv#scrollTo=TXtwTMfU-Hkw
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense , Dropout ,Flatten , MaxPooling2D
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Model, Sequential
import random
from keras.applications import MobileNetV2



mobile = MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')

#print(mobile.summary())
'''mobile = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(pool_size=(3,3)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(3,3)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3,3))])'''
# layer should not be change
for layer in mobile.layers:
  layer.trainable = False


# Make output layer of mobilenet
op_layer = mobile.output
op_layer = MaxPooling2D(pool_size=(3,3))(op_layer)
op_layer = Flatten()(op_layer)
op_final = Dense(256,activation='relu')(op_layer)
op_final = Dropout((0.5))(op_final)
op_final = Dense(1,activation= 'sigmoid')(op_final)


# Define model input and output
model = Model(inputs = mobile.input , outputs = op_final)

# compiling model
model.compile(optimizer = 'Adam', 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

#tryz = '/content/gdrive/My Drive/drowsiness detector/datasets/train/face_data/closed_eye'
#try2 = '/content/gdrive/My Drive/drowsiness detector/datasets/train/face_data/open_eye'
#try3 = '/content/gdrive/My Drive/drowsiness detector/datasets/train/face_data/yawn'
#try4 = '/content/gdrive/My Drive/drowsiness detector/datasets/train/face_data/no_yawn'
#lt1 = list(paths.list_images(tryz));print(len(lt1))
#lt2 = list(paths.list_images(try2));print(len(lt2))
#lt3 = list(paths.list_images(try3));print(len(lt3))
#lt4 = list(paths.list_images(try4));print(len(lt4))



dataset = '/content/gdrive/My Drive/drowsiness detector/datasets/train/face_data'

# initialize the initial learning rate, number of epochs to train for,
# and batch size
EPOCHS = 50
BS = 32

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))
random.seed(42)
random.shuffle(imagePaths)
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:

	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]
	# load the input image (150x150) and preprocess it
	image = load_img(imagePath, target_size=(224,224))
	image = img_to_array(image)/255.
	
 
	#image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
label_value = to_categorical(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=0,shuffle = True)

aug_train = ImageDataGenerator(rescale= 1.0/255.,
	rotation_range=20,
	zoom_range=0.15,
	zca_whitening=True,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

aug_test  = ImageDataGenerator(rescale= 1.0/255.)

def scheduler(epoch, lr):
   if epoch < 10:
     return lr
   else:
     return lr * tf.math.exp(-0.1)
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)]


hist = model.fit_generator(steps_per_epoch=len(trainX)//BS,
                           generator=aug_train.flow(trainX, trainY, batch_size=BS),
                           validation_data= (testX, testY),
                           validation_steps=len(testX)//BS,
                           callbacks=my_callbacks,
			   epochs=EPOCHS)

# print accuracy and loss graph
import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

# evaluating model on test data for loss and accuracy
acc = model.evaluate(testX,testY)
print('validation accuracy is {}'.format(acc[1]))
print('validation loss is {}'.format(acc[0]))

#model.save( 'model_face',overwrite=True)


model_new = Model(inputs = mobile.input , outputs = op_layer)
train_new = model_new.predict(trainX)
test_new = model_new.predict(testX)
print(test_new)

from sklearn.svm import SVC

svm = SVC(kernel='rbf')

svm.fit(train_new,trainY)
svm.score(train_new,trainY)
svm.score(test_new,testY)



