from nodeList import *
from random import randint
import numpy as np
import math as m
import os
import matplotlib.pyplot as plt
from PIL import Image


class Approximator:
  bounds = {"xBound":[-2,2],"yBound":[-15,15]}
  def __init__(self,func,samples = 20, Bounds = bounds):
    self.func = func
    self.samples = samples
    self.bounds = Bounds
    self.nodes = nodeList(self.samples,self.bounds["xBound"])
    self.x = np.linspace(self.bounds["xBound"][0],self.bounds["xBound"][1],samples+1)
    y=[]
    for x in self.x:
      y.append(self.func(x))
    self.answers = y

  def animate(self):
    self.gradientDescent()
    data = self.memory
    loss_data = self.total_loss
    newpath = "/Users/daniel/Desktop/Projects/PythonFolder/LinRegNeuralNetworkVisualization/images"
    if os.path.exists(newpath):
      os.system('rm -rf '+newpath)  #remove directory if exists
    os.makedirs(newpath)  #make a new directory
    index = 0
    for i in range (len(data)):
      fig,ax = plt.subplots()
      ax.set_xlim(self.bounds["xBound"][0],self.bounds["xBound"][1])
      ax.set_ylim(self.bounds["yBound"][0],self.bounds["yBound"][1])
      ax.plot(self.x,self.answers)
      ax.plot(self.x,data[i])
      plt.savefig((newpath+f"/{index}.png"))
      index+=1
      plt.close()
    images = [Image.open((newpath+f"/{n}.png")) for n in range(index)]
    images[0].save((newpath+"/Learn.gif"), save_all=True, append_images=images[1:], duration=100, loop=0)

  def gradientDescent (self,learningRate = .0125, iterations = 65,stop = 1):
    self.memory = []
    self.total_loss = []
    self.x_data = np.linspace(self.bounds["xBound"][0],self.bounds["xBound"][1],self.samples+1)
    for i in range(iterations):
      self.predicted_answers = [self.predict(x) for x in self.x_data]
      loss = [((self.answers[index]-self.predicted_answers[index])**2) for index in range(len(self.x_data))]
      if sum(loss)<stop:
        self.memory.append(self.predicted_answers)
        self.total_loss.append(sum(loss))
        break
      for node in self.nodes:
        dLdc = [self.change_c(index,node) for index in range(len(self.x_data))]
        dLda = [self.change_a(index,node) for index in range(len(self.x_data))]
        node.c -= learningRate*(sum(dLdc)/len(dLdc))
        node.a -= learningRate*(sum(dLda)/len(dLda))
      if i%1 == 0:
        self.memory.append(self.predicted_answers)
        self.total_loss.append(sum(loss))

  def predict(self,x):
    prediction = 0
    for node in self.nodes:
      prediction += node.c*m.sin(node.a*x)
    return prediction

  def pDeriv_a(self,c,a,x):
    return -1*c*x*m.cos(a*x)

  def pDeriv_c(self,c,a,x):
    return -1*m.sin(a*x)

  def change_c(self,index,node):
    x = self.x[index]
    p1 = 2*(self.answers[index]-self.predicted_answers[index])
    p1 *= self.pDeriv_c(node.c,node.a,x)
    return p1

  def change_a(self,index,node):
    x = self.x[index]
    p1 = 2*(self.answers[index]-self.predicted_answers[index])
    p1*=self.pDeriv_a(node.c,node.a,x)
    return p1
















