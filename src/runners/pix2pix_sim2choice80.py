import os
import sys
sys.path.append("../..")
sys.path.append("../../,,")
from scripts.pix2pix_sim2choice80 import pix2pix
current_device = 'cuda:0'
json_file_name = 'train_valid_test.json'
json_folder = '../config/train_valid_test'
json_path = os.path.join(json_folder, json_file_name)
log_file = "../../result/log/train_valid_test"
saved_model_file = "../../result/model"
batch_size = 24
import torch
torch.cuda.empty_cache()
pix2pix_model = pix2pix(current_device, batch_size, json_path, log_file,saved_model_file,\
                num_workers=None, lr=2e-4, beta=(0.5, 0.999), ori_size=(256,452))
pix2pix_model.train(epochs=50)