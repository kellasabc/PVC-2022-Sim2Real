import sys
sys.path.append("..")
sys.path.append("../..")
#import src
from src.model.models import pix2pix
import os
current_device = 'cuda:0'
json_file_name = 'train_valid_test.json'
json_folder = './result/json files/train_valid_test'
json_path = os.path.join(json_folder, json_file_name)
log_file = "./result/log/train_valid_test"
batch_size = 24
pix2pix_model = pix2pix(current_device, batch_size, json_path, log_file,\
                num_workers=None, lr=2e-4, beta=(0.5, 0.999), ori_size=(256,452))