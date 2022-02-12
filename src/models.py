import numpy as np
import pandas as pd

class DefaultModel:
   def __init__(self):
      self.model_type = "default"
      self.weights = np.array([0])
      self.train_data = np.array([0])

   def save(self,model_dir):
      '''
      Pickle up any data you wish to store to use for future purposes
      '''
      #np.save(model_dir + self.model_type + '.npy', self.weights, allow_pickle=True)
      self.weights = [] #filler code to be deleted and replaced by line above

   def load(self,model_dir):
      '''
      Load in stored data that has been previously trained
      '''
      self.weights = [] #np.load(model_dir + self.model_type + '.npy')

   def train(self,data_dir):
      '''
      Load in training and third party models to train on
      Method will be overloaded by derived classes
      '''
      self.train_data = [] #pd.read_csv("")

   def test(self,data_dir):
      '''
      Load in test data and third party models to test on
      Method will be overloaded by derived classes
      '''
      self.test_data = [] #pd.read_csv("")

class ConvexModel(DefaultModel):
   def __init__(self):
      DefaultModel.__init__(self);
      self.model_type = "convex"
   
   def train(self,data_dir):
      self.train_data = [] #pd.read_csv("")

   def test(self,data_dir):
      self.test_data = [] #pd.read_csv("")

def get_model(model_type):
   if "convex" == model_type:
      return ConvexModel()
   else:
      return DefaultModel()
