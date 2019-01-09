#!/usr/bin/python

import math
import numpy, random
import numpy as np
import sys

class NN:

       def __init__(self):

          bias_array = []
          self.bias_array =bias_array
          weights_array = []
          self.weights_array = []
      
       def biases(self, nl):

           self.nl = nl
           for i in range(1, len(nl)):
               biases = numpy.random.rand(nl[i])*0.01
               self.bias_array.append(biases)

           for i in range(0,len(nl)-1):
               weights = numpy.random.rand(nl[i],nl[i+1])*0.01
               self.weights_array.append(weights)

           return self.bias_array, self.weights_array



       def Activation(self, weights_matrix, biases_array, ins):
              act = []
              act = numpy.dot(ins, weights_matrix) + biases_array
              return act


       def activation_fn(self, Layer, act, flag_derivative):
       
         self.activation = numpy.ndarray((numpy.shape(act)[0]))
         if Layer  == "S":
            for i in range(0,numpy.shape(self.activation)[0]):

                  if flag_derivative == 0:
                     self.activation[i] = 1/(1+ math.exp(-act[i]))
                     self.activation = self.activation

                  else:
                    self.activation[i] = (act[i])*(1-act[i])

         elif  Layer == "N":
            for i in range(0,numpy.shape(self.activation)[0]):

                if flag_derivative == 0:

                  self.activation[i] =  math.tanh(act[i])
                  self.activation = self.activation
                else:
                    self.activation[i] = (1-act[i])*(1+act[i])

         elif  Layer == "softmax":

            s = 0
            for j in range(0,len(act)):
                  s = s + math.exp(act[j])
            for i in range(0,numpy.shape(self.activation)[0]):

               if flag_derivative == 0:
                  self.activation[i] = math.exp(act[i])/float(s)
                  self.activation = self.activation

               else:
                 for dd in range(0,len(act)):
                  if dd == i:
                   self.activation[i] = act[i]*(1-act[i])
                  else:
                   self.activation[i] = -(act[i]*act[dd])


         elif Layer == "L":

               for i in range(0,numpy.shape(self.activation)[0]):

                  if flag_derivative == 0:
                     self.activation[i] = act[i]
                  else:
                     self.activation[i] = 1

         return self.activation



       def forward_prop(self,ins, outs, outshape, error, momentum, lr, epoch):

           self.error = error
           self.ins = ins
           x = self.ins
           self.outs = outs
           self.momentum = momentum
           self.lr = lr
           self.epoch = epoch
           self.outshape = outshape

           if self.epoch > 10:
                 print("am reducing learning rate")
                 self.lr = self.lr/2
           else:
                pass

           nodeouts_array = []


           for i in range(1,len(self.nl)-1):
               node_outs = self.activation_fn("S",self.Activation(self.weights_array[i-1], self.bias_array[i-1],x),0)
               x = node_outs
               nodeouts_array.append(node_outs)

           node_outs = self.activation_fn("softmax",self.Activation(self.weights_array[len(self.weights_array)-1], self.bias_array[len(self.bias_array)]))
           nodeouts_array.append(node_outs)
           Error = self.error_calculation(nodeouts_array[len(nodeouts_array)-1])
           self.update_params(Error,self.weights_array, nodeouts_array)
           return Error

       def error_calculation(self,nodeouts):
           self.error = self.error + (self.outs - nodeouts)
           return self.error



# Data input

input = numpy.loadtxt("/home/siri/Documents/Projects/VC_2018/Data/database/input_sf1.txt")
output = numpy.loadtxt("dummy.txt")


# Data Normalisation

from sklearn import  preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
input =min_max_scaler.fit_transform(input)

# Data Preprocessing

from math import sqrt
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(input, output, train_size=0.83)
training_data = list(zip(Xtrain, Ytrain))
test_data = list(zip(Xtest, Ytest))

# NN architecture

n_layers = []
n_layers = [52, 100, 1]
epochs = 20
momentum = 0.9
learning_rate = 0.1

# Calling NN class

ss = NN()
biases, weights = ss.biases(n_layers)

print("biases and weights are:", numpy.shape(biases[0]), numpy.shape(biases[1]), numpy.shape(weights[0]), numpy.shape(weights[1]))

for j in range(0,epochs):
    error = 0
    random.shuffle(training_data)
   
    for p in range(0,len(training_data)):
      error = ss.forward_prop(training_data[p][0],training_data[p][1], n_layers[len(n_layers)-1],(error**2), momentum, learning_rate, j+1)
    print(sqrt((error)**2/len(training_data)))

