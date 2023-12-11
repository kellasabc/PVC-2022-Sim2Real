import sys
sys.path.append("../..")
sys.path.append("..")
from src.utils.utils import data_json
data_json_ = data_json(folder="../config")
data_json_.train_valid_test(train_radio=0.6, \
                       valid_radio=0.2, \
                       simulated_data_path='../../../../Code/praktikum/data/test/simulated', \
                       file_Name='train_valid_test.json', \
                       seed=1024)

data_json_.k_folder(seed =1024,test_radio = 0.2, \
                 simulated_data_path='../../Code/praktikum/data/test/simulated',\
                 k = 5)