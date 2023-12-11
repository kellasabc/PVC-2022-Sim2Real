import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import json
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import shutil
import sys
sys.path.append("../..")
from src.data.datasets_utils import Images_Dataset, Images_Data_Test
from src.utils.utils import check_points
from src.model.models import PatchGAN, Generator, _weights_init

class pix2pix():
    """
    class used to transfer picture from simulate to choice80
    """
    def __init__(self, current_device, batch_size, json_path, log_file, saved_model_file,\
                num_workers=None, lr=2e-4, beta=(0.5, 0.999), ori_size=(256,452), task="train_valid_test", k=0):
        """

        :param current_device: which device you want to use, i.g. "cuda:0"
        :param batch_size: depands on the memory of GPU, i.g. batch_siz=16 for 8GB memory GPU
        :param json_path: the path of json file, which have the path information of dataset for model,\
         i.g. './result/json files/train_valid_test''./result/json files/train_valid_test'
        :param log_file: the folder which you want to save the information of log
        :param saved_model_file: the folder which you want to save and load the trained model
        :param num_workers: parameter num_workers you want to pass to DataLoader
        :param lr: parameter learning_rate you want to pass to optimizor
        :param beta: parameter beta you want to pass to optimizor
        :param ori_size: the original size of input data, i.g. (256,452)
        :param task: whether you are working on "train_valid_test" or "k_folder"
        :param k: the K_th folder you are working on
        """
        self.current_device = current_device
        self.batch_size = batch_size
        self.log_file = log_file
        self.saved_model_file = saved_model_file
        self.json_path = json_path
        self.device = torch.device(self.current_device if (torch.cuda.is_available()) else 'cpu')
        self.lr = lr
        self.beta = beta
        self.ori_size = ori_size
        print("Availble devic of this machine is ---->  {}".format(self.device))
        with open(self.json_path, 'r', encoding='utf-8') as file:
            self.data_Info = json.load(file)
        if os.path.exists(self.log_file):
            shutil.rmtree(self.log_file)
        os.makedirs(self.log_file)
        self.writer = SummaryWriter(log_dir=self.log_file)
        train_dataset = Images_Dataset(self.data_Info, key_type='train')
        valid_dataset = Images_Dataset(self.data_Info, key_type='valid')
        if not num_workers:
            self.num_workers=0
        self.train_loader = DataLoader(train_dataset, \
                                  batch_size=self.batch_size, num_workers=self.num_workers)
        self.valid_loader = DataLoader(valid_dataset, \
                                  batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_dataset = None
        self.D = PatchGAN(6).to(device=self.device)
        self.G = Generator(3,3).to(device=self.device)
        self.G = self.G.apply(_weights_init)
        self.D = self.D.apply(_weights_init)

        self.BCE_loss = torch.nn.BCEWithLogitsLoss().to(device=self.device)
        self.L1_loss = torch.nn.L1Loss().to(device=self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=self.beta)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=self.beta)
        self.D_avg_losses = []
        self.G_avg_losses = []
        self.G_valid_avg_losses=[]
        self.D_valid_avg_losses=[]
        self.G_valid_avg_losses_min = 1e10
        if task=="train_valid_test":
            self.check_points = check_points(folder=os.path.join(self.saved_model_file,"train_valid_test"))
        else:
            self.check_points = check_points(folder=os.path.join(self.saved_model_file, "k_folder/folder_{}".format(k)))

    def train(self, epochs=100, start_epoch=0):
        """
        work on train phase of model
        :param epochs: how epochs you want to iterate
        :param start_epoch: set the start epoch for this iteration, i.e. if you have run epochs 100 before, then set \
        start_epoch as 100  when you want to run after it
        :return: None
        """
        print("*" * 30)
        print("train phase")
        print("*" * 30 + "\n")

        for epoch in range(start_epoch+1, epochs + 1):
            self.G.train()
            self.D.train()
            D_losses = []
            G_losses = []
            for imgs, labels in self.train_loader:
                imgs = imgs.to(device=self.device)  # <1>
                labels = labels.to(device=self.device)

                # Train discriminator with real data
                D_real_decision = self.D(imgs, labels)
                real_ = torch.ones_like(D_real_decision).to(device=self.device)
                D_real_loss = self.BCE_loss(D_real_decision, real_)

                # Train discriminator with fake data
                gen_image = self.G(imgs)
                D_fake_decision = self.D(imgs, gen_image)
                fake_ = torch.zeros_like(D_fake_decision).to(device=self.device)
                D_fake_loss =self.BCE_loss(D_fake_decision, fake_)

                # Back propagation
                D_loss = (D_real_loss + D_fake_loss) * 0.5
                self.D.zero_grad()
                D_loss.backward()
                self.D_optimizer.step()

                # Train generator
                gen_image = self.G(imgs)
                D_fake_decision = self.D(imgs, gen_image)
                G_fake_loss = self.BCE_loss(D_fake_decision, real_)

                # L1 loss
                l1_loss = 100 * self.L1_loss(gen_image, labels)

                # Back propagation
                G_loss = G_fake_loss + l1_loss
                self.G.zero_grad()
                G_loss.backward()
                self.G_optimizer.step()

                # loss values
                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())

            D_avg_loss = torch.mean(torch.Tensor(D_losses))
            G_avg_loss = torch.mean(torch.Tensor(G_losses))

            # avg loss values for plot
            self.D_avg_losses.append(D_avg_loss)
            self.G_avg_losses.append(G_avg_loss)

            if epoch == 1 or epoch % 10 == 0:
                print('{} Epoch {}, Generator loss {}, Discriminator loss {}'.format(
                    datetime.datetime.now(), epoch,
                    G_avg_loss, D_avg_loss))
                self.valid(cur_epoch=epoch)
                if self.G_valid_avg_losses[-1] < self.G_valid_avg_losses_min:
                    self.G_valid_avg_losses_min  = self.G_valid_avg_losses[-1]
                    print("the model is saved")
                    self.check_points.save(epoch,self.G,self.D,self.G_optimizer,self.D_optimizer)
                fig, axes = plt.subplots(1, 3, figsize=(5, 5))
                imgs = [imgs[-1], gen_image[-1], labels[-1]]
                show_imgs = []
                for ax, img in zip(axes.flatten(), imgs):
                    ax.axis('off')
                    ax.set_adjustable('box')
                    # Scale to 0-255
                    img = (((img - img.min()) * 255) / (img.max() - img.min())).detach().cpu().numpy().transpose(1, 2,
                                                                                                                 0).astype(
                        np.uint8)
                    show_imgs.append(img)
                    ax.imshow(img, cmap=None, aspect='equal')
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.show()
                self.writer.add_scalars('train', {'Generator loss': G_avg_loss,
                                             'Discriminator loss': D_avg_loss}, epoch)

                self.writer.add_image('inpute image', show_imgs[0], epoch, dataformats='HWC')
                self.writer.add_image('target image', show_imgs[2], epoch, dataformats='HWC')
                self.writer.add_image('generated image', show_imgs[1], epoch, dataformats='HWC')


    def valid(self, cur_epoch):
        """
        works on valid phase of model
        :return: None
        """
        G_losses = []
        D_losses = []
        print("*" * 30)
        print(" validation of epoch {}".format(cur_epoch))
        print("*" * 30 + "\n")
        self.G.eval()
        with torch.no_grad():
            for imgs, labels in self.valid_loader:
                imgs = imgs.to(device=self.device)  # <1>
                labels = labels.to(device=self.device)

                # validate discriminator with real data
                gen_image = self.G(imgs)
                D_real_decision = self.D(imgs, labels)
                real_ = torch.ones_like(D_real_decision).to(device=self.device)
                D_real_loss = self.BCE_loss(D_real_decision, real_)
                labels = labels.to(device=self.device)

                D_fake_decision = self.D(imgs, gen_image)
                fake_ = torch.zeros_like(D_fake_decision).to(device=self.device)
                D_fake_loss = self.BCE_loss(D_fake_decision, fake_)

                # Back propagation
                D_loss = (D_real_loss + D_fake_loss) * 0.5


                # validate generator
                D_fake_decision = self.D(imgs, gen_image)
                G_fake_loss = self.BCE_loss(D_fake_decision, real_)

                # L1 loss
                l1_loss = 100 * self.L1_loss(gen_image, labels)

                # Back propagation
                G_loss = G_fake_loss + l1_loss
                D_losses.append(D_loss.item())

                G_losses.append(l1_loss.item())


            G_avg_loss = torch.mean(torch.Tensor(G_losses))
            D_avg_loss = torch.mean(torch.Tensor(D_losses))

            # avg loss values for plot
            self.G_valid_avg_losses.append(G_avg_loss)
            self.D_valid_avg_losses.append(D_avg_loss)
            print('{} Epoch {}, Generator loss {}, Discriminator loss {}'.format(
                datetime.datetime.now(), cur_epoch,
                G_avg_loss, D_avg_loss))
            fig, axes = plt.subplots(1, 3, figsize=(5, 5))
            imgs = [imgs[-1], gen_image[-1], labels[-1]]
            show_imgs = []
            for ax, img in zip(axes.flatten(), imgs):
                ax.axis('off')
                ax.set_adjustable('box')
                # Scale to 0-255
                img = (((img - img.min()) * 255) / (img.max() - img.min())).detach().cpu().numpy().transpose(1, 2,
                                                                                                             0).astype(
                    np.uint8)
                show_imgs.append(img)
                ax.imshow(img, cmap=None, aspect='equal')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()
            self.writer.add_scalars('valid', {'Generator loss': G_avg_loss,
                                              'Discriminator loss': D_avg_loss}, cur_epoch)

            self.writer.add_image('inpute image_valid ', show_imgs[0], cur_epoch, dataformats='HWC')
            self.writer.add_image('target image_valid', show_imgs[2], cur_epoch, dataformats='HWC')
            self.writer.add_image('generated image_valid', show_imgs[1], cur_epoch, dataformats='HWC')

    def test(self, save_path = './result/generated test images'):
        """
        work on test phase of model: generate image and monitor image in test phase
        :param save_path: the path you want to save these generated images in test phase
        :return: None
        """
        print("*" * 30)
        print("test phase")
        print("*" * 30 + "\n")
        self.G.eval()
        G_losses = []
        self.test_dataset = Images_Data_Test(self.data_Info['test'])
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        monitor_path = save_path.replace('generated test images', 'monitor')
        if os.path.exists(monitor_path):
            shutil.rmtree(monitor_path)
        os.mkdir(monitor_path)
        with torch.no_grad():
            for ori_path, img, target in self.test_dataset:
                img = img.to(device=self.device)  # <1>
                splitted_ori_path = ori_path.split('\\')[1:]
                head = save_path
                for item in splitted_ori_path:
                    head = os.path.join(head, item)
                path = head.replace('inputs', 'style_02')

                target = target.to(device=self.device)
                gen_image = transforms.Resize((self.ori_size))(self.G(img.unsqueeze(0)).squeeze())

                # L1 loss
                l1_loss = 100 * self.L1_loss(gen_image.unsqueeze(0), target.unsqueeze(0))

                G_losses.append(l1_loss.item())
                self.writer.add_text("test", ' {}, L1 loss {}'.format( \
                    ori_path, l1_loss.item()))
                print(' {}, L1 loss {}'.format(ori_path, l1_loss.item()))
                # Scale to 0-255
                generated_img = (((gen_image - gen_image.min()) * 255) / (gen_image.max() - gen_image.min())). \
                    detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                target_img = (((target - target.min()) * 255) / (target.max() - target.min())). \
                    detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                ori_img = np.array(Image.open(ori_path))
                filepath, _ = os.path.split(path)
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                Image.fromarray(generated_img).save(path, "PNG")
                monitor_path = path.replace('generated test images', 'monitor')
                filepath_monitor, _ = os.path.split(monitor_path)
                if not os.path.exists(filepath_monitor):
                    os.makedirs(filepath_monitor)
                monitor_img = np.concatenate((ori_img, generated_img, target_img), axis=1)
                Image.fromarray(monitor_img).save(monitor_path, "PNG")
        print("the average loss of test is {}".format(torch.mean(torch.Tensor(G_losses))))
        print("*" * 30)
        print("test phase finished!")
        print("*" * 30 + "\n")

    def load_model(self,epoch_num):
        """
        load the saved data point to model
        :param epoch_num: from epoch you want to load
        :return: None
        """
        file_name = "epoch_{}.tar".format(epoch_num)
        cur_checkpoints = self.check_points.load(file_name)
        self.G.load_state_dict(cur_checkpoints['generator'])
        self.load_state_dict(cur_checkpoints['discriminator'])
        self.G_optimizer.load_state_dict(cur_checkpoints['G_optimizer'])
        self.D_optimizer.load_state_dict(cur_checkpoints['D_optimizer'])
        self.D_avg_losses = []
        self.G_avg_losses = []
        self.G_valid_avg_losses = []

    def save(self,cur_epoch):
        self.check_points.save(cur_epoch, self.G, self.D, self.G_optimizer, self.D_optimizer)