import models

from config import configure

configs = configure()

def train():
   print("Training with model: " + configs.model_type)
   model = models.get_model(configs.model_type)
   model.train(configs.train_data_dir)
   model.save(configs.model_dir)
   

if __name__ == "__main__":
   train()
