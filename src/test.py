import models

from config import configure

configs = configure()

def test():
   print("Testing with model: " + configs.model_type)
   model = models.get_model(configs.model_type)
   model.load(configs.model_dir)
   model.test(configs.test_data_dir);

if __name__ == "__main__":
   test()
