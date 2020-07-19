
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical



global shape_tracker
shape_tracker=[]

#Loading dataset.
(X_train,y_train),(X_test,y_test)=mnist.load_data()

#Reshaping into 784 dimensional array.
X_train=X_train.reshape(-1,784)
X_test=X_test.reshape(-1,784)

X_train=X_train/255
X_test=X_test/255

y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)


class Softmax:
	def __init__(self):
		pass
	def feed_forward(self,x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

def add2trackerdict(val):
    shape_tracker.append(val)




class Dense:
    def __init__(self,input_dim=False,n_units=0,learning_rate=0.1):

        if input_dim:
            self.learning_rate=learning_rate
            self.weights=np.random.randn(input_dim,n_units)*0.01
            self.bias=np.zeros(n_units)
            add2trackerdict(n_units)
        else:
            self.learning_rate=learning_rate
            self.weights=np.random.randn(shape_tracker[-1],n_units)*0.01
            self.bias=np.zeros(n_units)
            add2trackerdict(n_units)

    def feed_forward(self,input):
        return np.matmul(input,self.weights)+self.bias

class ReLU:
    def __init__(self):
        pass
    def feed_forward(self,x):
        return np.maximum(0,x)





class Sequential:
    def __init__(self):
         self.model=[]


    def add(self,obj):
        if len(shape_tracker)==0:
            raise Exception("Input shape not specified.")
        else:
            self.model.append(obj)

    def feed_forward(self,X):
        fpassstack=[]
        input=X
        for i in range(len(self.model)):
            fpassstack.append(self.model[i].feed_forward(X))
            X=self.model[i].feed_forward(X)
        return fpassstack

    def predict(self,X):
        logits=self.feed_forward(X)[-1]
        return logits.argmax(axis=-1)




model=Sequential()
model.add(Dense(input_dim=X_train.shape[1],n_units=32))
model.add(ReLU())
model.add(Dense(n_units=10))
model.add(Softmax())







trained_model=load_model('testmodel.h5')

W1=trained_model.layers[0].get_weights()[0]
W2=trained_model.layers[1].get_weights()[0]

model.model[0].weights=W1
model.model[2].weights=W2

print(model.predict(X_test[0]))

plt.imsave('digit.jpg',X_test[0].reshape(28,28))

