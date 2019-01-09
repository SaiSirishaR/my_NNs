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

      def biases(self, nl):

           self.nl = nl
           for i in range(1, len(nl)):
               biases = numpy.random.rand(nl[i])*0.01
               self.bias_array.append(biases)

           for i in range(0,len(nl)-1):
               weights = numpy.random.rand(nl[i],nl[i+1])*0.01
               self.weights_array.append(weights)

      def Activation(self, weights_matrix, biases_array, ins):
              act = []
              act = numpy.dot(ins, weights_matrix) + biases_array
              return act

       def update_params(self,Error,weights, nodeouts_array):

              new_bias_array = []
              new_weights_array = []

              grad_outputs =  Error * self.activation_fn("softmax",nodeouts_array[len(nodeouts_array)-1],1)
              new_bias = self.lr*grad_outputs

              if self.outshape == 1:

               if  self.previous_deltas == []:
                 print "sss"
                 output_deltas = ((nodeouts_array[len(nodeouts_array)-2])*grad_outputs)
               else:
                 print "am adding momentum at o/p layer"
                 output_deltas = ((nodeouts_array[len(nodeouts_array)-2])*grad_outputs)*self.previous_deltas[0]*self.momentum
               self.previous_deltas.append(output_deltas)
               output_deltas = np.expand_dims(output_deltas, axis=0)
               new_weight_matrix = self.lr*(numpy.transpose(weights[len(nodeouts_array)-1])+output_deltas)*nodeouts_array[len(nodeouts_array)-2]
               new_weights_array.append(numpy.transpose(new_weight_matrix))

              else:
                if len(self.previous_deltas) == 0:
                 output_deltas = np.outer((nodeouts_array[len(nodeouts_array)-2]), grad_outputs)
                else:
                 output_deltas = np.outer((nodeouts_array[len(nodeouts_array)-2]), grad_outputs)*self.previous_deltas[0]*self.momentum
              self.previous_deltas.append(output_deltas)
              new_weight_matrix = self.lr*((weights[len(nodeouts_array)-1])+output_deltas)*nodeouts_array[len(nodeouts_array)-2]
              new_weights_array.append((new_weight_matrix))

              new_bias_array.append(new_bias)

              i =  len(nodeouts_array)-2


              for j in range(0, len(nodeouts_array)-2):
                 print "j is", j
                 derivative = (self.activation_fn("N", nodeouts_array[i], 1))
                 grad_hidden = derivative*numpy.transpose(numpy.dot(weights[i+1], numpy.transpose(grad_outputs)))
                 new_bias = self.lr * grad_hidden
                 if len(self.previous_deltas) < len(weights)-1:
                    deltas_hidden = np.outer((nodeouts_array[i-1]), grad_hidden)
                 else:
                    print "am adding momentum at hidden layer num", j
                    deltas_hidden = np.outer((nodeouts_array[i-1]), grad_hidden)*self.previous_deltas[j+1]*self.momentum
                 self.previous_deltas.append(deltas_hidden)
                 grad_outputs = grad_hidden
                 new_weights = np.outer(self.lr * (weights[i]+deltas_hidden),nodeouts_array[j-1])
                 new_bias_array.append(new_bias)
                 new_weights_array.append(new_weights)
                 i = i -1


              new_bias_h1 = self.lr*(nodeouts_array[0])
              delta_weights = np.outer((nodeouts_array[i-1]), grad_hidden)*self.previous_deltas[j+1]*self.momentum
              new_weights_i_h = self.lr*delta_weights*np.outer(self.ins,nodeouts_array[0])
              new_weights_array.append(new_weights_i_h)
              new_bias_array.append(new_bias_h1)

              New_weights_array = []
              New_biases_array = []
              p = len(new_weights_array)-1

              for i in range(0,len(new_weights_array)):
                  New_weights_array.append(new_weights_array[p])
                  New_biases_array.append(new_bias_array[p])
                  p = p -1

              self.bias_array = New_biases_array
              self.weights_array = New_weights_array
              numpy.save('best_weights.npy', New_weights_array)
              numpy.save('best_biases.npy', New_biases_array)


       def forward_prop(self,ins, outs, outshape, error, momentum, lr, epoch, previous_deltas):
           self.error = error
           self.ins = ins
           x = self.ins
           self.outs = outs
           self.momentum = momentum
           self.lr = lr
           self.epoch = epoch
           self.previous_deltas = previous_deltas
           self.outshape = outshape

           if self.epoch > '10':
                 print "am reducing learning rate"
                 self.lr = self.lr/2
           else:
                self.lr = lr

           nodeouts_array = []


           for i in range(1,len(self.nl)-1):
               node_outs = self.activation_fn("N",self.Activation(self.weights_array[i-1], self.bias_array[i-1],x),0)
               x = node_outs
               nodeouts_array.append(node_outs)

           node_outs = self.activation_fn("softmax",self.Activation(self.weights_array[len(self.weights_array)-1], self.bias_array[len(self.bias_arr$
           nodeouts_array.append(node_outs)
           Error = self.error_calculation(nodeouts_array[len(nodeouts_array)-1])
           self.update_params(Error,self.weights_array, nodeouts_array)
           return Error

       def error_calculation(self,nodeouts):
           self.error = self.error + (self.outs - nodeouts)
           return self.error


input = numpy.loadtxt("/home/siri/Documents/Projects/VC_2018/Data/database/input_sf1.txt")
output = numpy.loadtxt("dummy.txt")
n_layers = []
n_layers = [4,15,20,25,10,1]
epochs = 20
momentum = 0.9
learning_rate = 0.1
training_data = []
ss = NN()
ss.biases(n_layers)
previous_deltas = []

from sklearn import  preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
input =min_max_scaler.fit_transform(input)
output =min_max_scaler.fit_transform(output)


from math import sqrt
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(input, output, train_size=0.83)
training_data = zip(Xtrain, Ytrain)
test_data = zip(Xtest, Ytest)


for j in range(0,epochs):
  error = 0
  random.shuffle(training_data)
  print "epoch num", j+1
  if j+1 == 1:
    for i in range(0,len(training_data)):
      print "input num is", i
      error = ss.forward_prop(training_data[i][0],training_data[i][1], n_layers[len(n_layers)-1],(error**2), 1, learning_rate, j+1, previous_deltas)
    print sqrt((error)**2/len(training_data))
  else:
    for i in range(0,len(training_data)):
      print "input num is", i
      error = ss.forward_prop(training_data[i][0],training_data[i][1], n_layers[len(n_layers)-1],(error**2), momentum, learning_rate, j+1, previous_deltas)
    print sqrt((error)**2/len(training_data))



