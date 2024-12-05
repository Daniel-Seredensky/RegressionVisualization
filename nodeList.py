from node import *
import numpy as np
class nodeList(node):
  def __init__(self,length,bounds):
    self.bounds = bounds
    self.index = 0
    self.length = length
    self.generate()

  def __getitem__(self,index):
    return self.nList[index]

  def __len__(self):
    return len(self.nList)

  def __iter__(self):
    return self

  def __next__(self):
    try:
      x = self.nList[self.index]
    except IndexError:
      self.index = 0
      raise StopIteration
    self.index += 1
    return x

  def generate(self):
    self.nList = [node() for index in np.linspace(self.bounds[0],self.bounds[1],self.length+1)]
    return self.nList

