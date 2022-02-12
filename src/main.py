from config import configure
from test import test
from train import train

configs = configure()

def main():
   if (configs.train):
      train()

   if (configs.test):
      test()
      
if __name__ == "__main__":
   main()
