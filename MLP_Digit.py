#Importing the necessary libraries and the dataset.
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import to_categorical
from keras.models import load_model

#Loading dataset.
(X_train,y_train),(X_test,y_test)=mnist.load_data()

#Reshaping into 784 dimensional array.
X_train=X_train.reshape(-1,784)
X_test=X_test.reshape(-1,784)

X_train=X_train/255
X_test=X_test/255

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


#Building and training the model.
model=Sequential()
model.add(Dense(32,activation='relu',input_shape=(784,)))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,steps_per_epoch=1000,epochs=2,validation_data=(X_test,y_test),validation_steps=400,shuffle=True)

model.save('testmodel.h5')



