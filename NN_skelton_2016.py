#!/usr/bin/python

import math
import numpy
import numpy as np
import sys

class NN:

       def __init__(self):

          bias_array = []
          self.bias_array = bias_array
          weights_array = []
          self.weights_array = []

       def activation_fn(self, Layer, act, flag_derivative):
 
          self.activation = numpy.adarray((numpy.shape(act[0])))

          if Layer == "S":

           for i in range(0,numpy.shape(self.activation[0])):
             self.activation[i] = 1/(1+math.exp(-act[i]))

             if flag_derivative == 0:
                self.activation = self.activation
             else:
                self.activation[i] = self.activation[i] * (1-self.activation[i])
           return self.activation          

     
          elif Layer == "N":

             for i in range(0,numpy.shape(self.activation[0])):
                self.activation[i] = math.tanh(act[i])

                if flag_derivative == 0:
                  self.activation = self.activation
                else:
                  self.activation[i] = (1-self.activation[i]) * (1+ self.activation[i])
             return self.activation


       def biases(self,nl):

           self.nl = nl
           for i in range(1,len(nl)):
               biases =numpy.random.rand(nl[i],nl[i+1])
               self.weights_array.append(weights) 


       def Activation(self, weights_matrix, biases_array, ins):

           act = []
           act = numpy.dot(ins, weights_matrix) + biases_array
           return act

       def update_params(self, Error, weights, nodeouts_array):
        
           new_bias_array = []
           new_weights_array = []
           lr = 0.01
           grad_outputs = Error * self.activation_fn("N", nodeouts_array[len(nodeouts_array)-1],1)
           new_bias = lr*grad_outputs
           output_deltas = np.expand_dims(output_deltas, axis = 0)
           new_weight_matrix = lr*(numpy.transpose(weights[len(nodeouts_array)-1])+ output_deltas)
           new_weights_array.append(numpy.transpose(new_weights_matrix))
           new_bias_array.append(new_bias) 

           i = len(nodeouts_array) - 2
           for j in range(0,len(nodeouts_array)-2):
               derivative = (self.activation_fn("N", nodeouts_array[i],1))
               grad_hidden = derivative*numpy.transpose(numpy.dot(weights[i+1],numpy.transpose(grad_outputs)))
               new_bias = lr*grad_hidden
               deltas_hidden = np.outer((nodeouts_array[i-1],grad_hidden)
               grad_outputs =  grad_hidden
               new_weights = lr*(weights[i]+delatas_hidden)
               new_bias_array.append(new_bias)
               new_weights_array.append(new_weights)
               i = i-1
           new_bias_h1 = lr*(nodeouts_array[0])
           new_weights_i_h = np.outer(self.ins, nodeouts_array[0])
           new_weights_array.append(new_weights_i_h)
           new_bias_array.append(new_bias_h1)

           New_weights_array = []
           New_biases_array = []
           p = len(new_weights_array)-1
           for i in range(0,len(new_weights_array)):
            New_weights_array.append(new_weights_array[p])
            New_biases_array.append(new_bias_array[p])
            p = p-1
           self.bias_array =New_biases_array
           self.weights_array = New_weights_array

       def forward_prop(self, ins, outs, epochs):

          self.ins = ins
          x = self.ins
          self.outs = outs
          nodeouts_array = []
          for i in range(1,len(self.nl)):
           node_outs = self.activation_fn("N", self.Activation(self.weights_array[i-1],self.bias_array[i-1],x),0)
           x = node_outs
           nodeouts_array.append(node_outs)
         Error = self.error_calculation(nodeouts_array[len(nodeouts_array)-1])

         for i in range(0,epochs):

          self.update_params(Error, self.weights_array, nodeouts_array)
         return nodeouts_array

       def error_calculation(self, nodeouts):
           error = self.outs-nodeouts
           return error
            
                        
input = numpy.loadtxt("/home/siri/Documents/Projects/VC_2018/Data/database/input_sf1.txt")
output = numpy.loadtxt("dummy.txt")
n_layers = []
epochs = 20

ss =NN()
ss.biases(n_layers)
for i in range(0,len(input)):
  ins = input[i]
  outs = output[i]
  node_outputs = ss.forward_prop(ins, outs, epochs)
