class Configuration():
   def __init__(self):
      self.model_type = "default"
      self.model_dir = "../data/model_data/"
      self.train = True
      self.train_data_dir = "../data/train/"
      self.test = True
      self.test_data_dir = "../data/test/"

def configure():
   return Configuration();
      
