"""
train cholec80 channel and depth channel togother, use skeleton model structure
try l2 loss(mean)
base on depth_3
try to add segmentation information into it
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "True"
import torch
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
import torchvision
import cv2
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
from src.data.datasets_utils_cholec_depth_seg import Cholec_Depth_Seg_Dataset, get_normalize_parameter, Images_Cholec_Depth_Seg_Test
from src.utils.utils import check_points
from src.model.models_different_head import PatchGAN, Generator, _weights_init
import os
class pix2pix():
    """
    class used to transfer picture from simulate to choice80, depth, segmentation
    """
    def __init__(self, current_device, batch_size, json_path, log_file,saved_model_file,\
                num_workers=None, lr=2e-4, beta=(0.5, 0.999), ori_size=(256,452), task="train_valid_test", k=0,\
                 radio=(50,500,250,1, 0.99, 0),dis_step=1):
        """

        :param current_device: which device you want to use, i.g. "cuda:0"
        :param batch_size: depands on the memory of GPU, i.g. batch_siz=16 for 8GB memory GPU
        :param json_path: the path of json file, which have the path information of dataset for model,\
         i.g. './result/json files/train_valid_test''./result/json files/train_valid_test'
        :param log_file: the folder which you want to save the information of log
        :param num_workers: parameter num_workers you want to pass to DataLoader
        :param lr: parameter learning_rate you want to pass to optimizor
        :param beta: parameter beta you want to pass to optimizor
        :param ori_size: the original size of input data, i.g. (256,452)
        :param task: whether you are working on "train_valid_test" or "k_folder"
        :param k: the K_th folder you are working on
        :param radio: self.ratio_dis, self.ratio_cholec, self.ratio_depth, self.ratio_seg, self.ratio_one, self.ratio_zero
        """
        print(os.getcwd())
        self.current_device = current_device
        self.batch_size = batch_size
        self.log_file = log_file
        self.saved_model_file = saved_model_file
        self.json_path = json_path
        self.device = torch.device(self.current_device if (torch.cuda.is_available()) else 'cpu')
        self.lr = lr
        self.beta = beta
        self.ori_size = ori_size
        self.ratio_dis, self.ratio_cholec, self.ratio_depth, self.ratio_seg,\
        self.ratio_one, self.ratio_zero, self.metrics = radio
        self.dis_step = dis_step
        print("Availble devic of this machine is ---->  {}".format(self.device))
        with open(self.json_path, 'r', encoding='utf-8') as file:
            self.data_Info = json.load(file)
        self.norm_param, self.seg_l2n, self.seg_n2l = get_normalize_parameter(self.data_Info)

        if os.path.exists(self.log_file):
            shutil.rmtree(self.log_file)
        os.makedirs(self.log_file)
        self.writer = SummaryWriter(log_dir=self.log_file)
        train_dataset = Cholec_Depth_Seg_Dataset(self.data_Info, self.seg_l2n,key_type='train')
        valid_dataset = Cholec_Depth_Seg_Dataset(self.data_Info, self.seg_l2n,key_type='valid')
        if not num_workers:
            self.num_workers = 0
        else:
            self.num_workers = int(num_workers)
        self.train_loader = DataLoader(train_dataset, \
                                  batch_size=self.batch_size, num_workers=self.num_workers)
        self.valid_loader = DataLoader(valid_dataset, \
                                  batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_dataset = None
        self.D = PatchGAN(8).to(device=self.device)
        self.G = Generator(3,3,1,len(self.seg_l2n)).to(device=self.device)
        self.G = self.G.apply(_weights_init)
        self.D = self.D.apply(_weights_init)

        self.BCE_loss = torch.nn.BCEWithLogitsLoss().to(device=self.device)
        self.L1_loss = nn.MSELoss().to(device=self.device)
        self.Cross_loss = nn.CrossEntropyLoss().to(device=self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=self.beta)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=self.beta)
        self.D_avg_losses = []
        self.G_avg_losses = []
        self.G_valid_avg_losses=[]
        self.D_valid_avg_losses=[]
        self.G_valid_avg_losses_min = 1e10
        if task=="train_valid_test":
            self.check_points = check_points(folder=os.path.join(self.saved_model_file, "train_valid_test"))
        else:
            self.check_points = check_points(folder=os.path.join(self.saved_model_file, "train_valid_test"))

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

        for epoch in range(start_epoch+1, epochs + start_epoch+ 1):
            self.G.train()
            self.D.train()
            D_losses = []
            G_losses = []
            for imgs, labels in self.train_loader:
                imgs = imgs.to(device=self.device)  # <1>
                labels = labels.to(device=self.device)
                for _ in range(self.dis_step):
                    # Train discriminator with real data
                    # Normalize the input image to range 0-1
                    # Normalize the 0-3 channel of traget to range 0-1
                    imgs_ = imgs.clone()
                    labels_ = labels.clone()
                    for channel in range(3):
                        imgs_[:,channel, :, :] = (imgs_[:,channel, :, :] - self.norm_param[0,channel,0]) * self.metrics/ \
                                               (self.norm_param[0,channel,1] - self.norm_param[0,channel,0])
                        labels_[:,channel, :, :] =  (labels_[:,channel, :, :] - self.norm_param[1,channel,0]) / \
                                               (self.norm_param[1,channel,1] -
                                                self.norm_param[1,channel,0])
                    channel = 3
                    labels_[:,channel, :, :] = (labels_[:,channel, :, :] - self.norm_param[1, channel, 0]) * self.metrics / \
                                             (self.norm_param[1, channel, 1] -
                                              self.norm_param[1, channel, 0])

                    D_real_decision = self.D(imgs_, labels_)
                    real_ = torch.ones_like(D_real_decision).to(device=self.device) * self.ratio_one
                    D_real_loss = self.BCE_loss(D_real_decision, real_)

                    # Train discriminator with fake data
                    gen_cholec80, gen_depth, gen_seg = self.G(imgs_)

                    gen_seg_ = torch.argmax(gen_seg,dim=1,keepdim=True)
                    gen_image = torch.cat((gen_cholec80, gen_depth, gen_seg_), axis=1)
                    gen_image_ = gen_image.clone()
                    for channel in range(3):
                        gen_image_[:,channel, :, :] = (gen_image_[:,channel, :, :] - self.norm_param[1,channel,0]) * self.metrics / \
                                                 (self.norm_param[1,channel,1] -
                                                  self.norm_param[1,channel,0])
                    D_fake_decision = self.D(imgs_, gen_image_)
                    fake_ = (torch.zeros_like(D_fake_decision)+ self.ratio_zero).to(device=self.device)
                    D_fake_loss =self.BCE_loss(D_fake_decision, fake_)

                    # Back propagation
                    D_loss = (D_real_loss + D_fake_loss) * 0.5 * self.ratio_dis
                    self.D.zero_grad()
                    D_loss.backward()
                    self.D_optimizer.step()

                # Train generator
                gen_cholec80, gen_depth, gen_seg = self.G(imgs_)
                gen_seg_ = torch.argmax(gen_seg, dim=1,keepdim=True)
                gen_image = torch.cat((gen_cholec80, gen_depth, gen_seg_), axis=1)
                gen_image_ = gen_image.clone()

                D_fake_decision = self.D(imgs_, gen_image_)
                G_fake_loss = self.BCE_loss(D_fake_decision, real_)

                # L1 loss
                l1_loss_cholec_80 = self.ratio_cholec * self.L1_loss(gen_image_[:, :3, :, :], labels_[:, :3, :, :])
                l1_loss_depth = self.ratio_depth * self.L1_loss(gen_image_[:, 3:4, :, :], labels_[:, 3:4, :, :])
                cross_loss_seg = self.ratio_seg * self.Cross_loss(gen_seg[:, :, :, :], labels_[:, 4, :, :].\
                                                       to(torch.long))
                l1_loss = l1_loss_cholec_80 + l1_loss_depth + cross_loss_seg
                self.writer.add_scalars('monitor train loss', {'loss of cholec80': l1_loss_cholec_80,\
                                                               'loss of depth': l1_loss_depth,\
                                                               'loss of seg': cross_loss_seg,}, epoch)

                # Back propagation
                G_loss = G_fake_loss * self.ratio_dis + l1_loss
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
                print('{} Epoch {}, Generator tain loss {}, Discriminator train loss {}'.format(
                    datetime.datetime.now(), epoch,
                    G_avg_loss, D_avg_loss))
                self.valid(cur_epoch=epoch)
                """if self.G_valid_avg_losses[-1] < self.G_valid_avg_losses_min:
                    self.G_valid_avg_losses_min  = self.G_valid_avg_losses[-1]
                    print("the model is saved")"""
                self.check_points.save(epoch,self.G,self.D,self.G_optimizer,self.D_optimizer)
                fig, axes = plt.subplots(1, 7, figsize=(5, 5))
                imgs = [imgs[-1], gen_image[-1][:3,:,:], labels[-1][:3,:,:], \
                        gen_image[-1][3:4,:,:], labels[-1][3:4,:,:],\
                        gen_image[-1][4:,:,:], labels[-1][4:,:,:]]
                imgs_key = ['input', 'generate_cholec80', 'cholec80',\
                            "generate_depth", 'depth',\
                            "generate_seg", "seg"]
                show_imgs = []

                for key, ax, img in zip(imgs_key ,axes.flatten(), imgs):
                    ax.axis('off')
                    ax.set_adjustable('box')
                    # Scale to 0-255

                    if key=="generate_depth":
                        img[0, :, :] = img[0, :, :] *\
                                             (self.norm_param[1,3,1] - self.norm_param[1,3,0])\
                                             + self.norm_param[1,3,0]
                    if key in ['input', 'generate_cholec80', 'cholec80',"generate_depth", 'depth']:
                        img = (((img - img.min()) * 255) / (img.max() - img.min())).detach().cpu() \
                            .numpy().transpose(1, 2, 0).astype(np.uint8)
                    if key in ["generate_seg", "seg"]:
                        n2l_f = lambda x: self.seg_n2l[x]
                        img = img.detach().cpu().to(torch.long).apply_(n2l_f)\
                            .numpy().transpose(1, 2, 0).astype(np.uint8)
                    #img = img.detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                    show_imgs.append(img)
                    ax.imshow(img, cmap=None, aspect='equal')
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.show()
                self.writer.add_scalars('train', {'Generator loss': G_avg_loss,
                                             'Discriminator loss': D_avg_loss}, epoch)

                self.writer.add_image('train_inpute image', show_imgs[0], epoch, dataformats='HWC')
                self.writer.add_image('train_cholec80 image', show_imgs[2], epoch, dataformats='HWC')
                self.writer.add_image('train_generated cholec80 image', show_imgs[1], epoch, dataformats='HWC')
                self.writer.add_image('train_depth image', show_imgs[4], epoch, dataformats='HWC')
                self.writer.add_image('train_generate depth image', show_imgs[3], epoch, dataformats='HWC')
                self.writer.add_image('train_seg image', show_imgs[6], epoch, dataformats='HWC')
                self.writer.add_image('train_generate seg image', show_imgs[5], epoch, dataformats='HWC')


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
                imgs_ = imgs.clone()
                labels_ = labels.clone()
                for channel in range(3):
                    imgs_[:, channel, :, :] = (imgs_[:, channel, :, :] - self.norm_param[0, channel, 0]) * self.metrics / \
                                              (self.norm_param[0, channel, 1] - self.norm_param[0, channel, 0])
                    labels_[:, channel, :, :] = (labels_[:, channel, :, :] - self.norm_param[1, channel, 0]) * self.metrics / \
                                                (self.norm_param[1, channel, 1] -
                                                 self.norm_param[1, channel, 0])
                channel = 3
                labels_[:, channel, :, :] = (labels_[:, channel, :, :] - self.norm_param[1, channel, 0]) * self.metrics / \
                                            (self.norm_param[1, channel, 1] -
                                             self.norm_param[1, channel, 0])
                # validate discriminator with real data

                gen_cholec80, gen_depth, gen_seg = self.G(imgs_)
                gen_seg_ = torch.argmax(gen_seg, dim=1, keepdim=True)
                gen_image = torch.cat((gen_cholec80, gen_depth, gen_seg_), axis=1)
                gen_image_ = gen_image.clone()
                for channel in range(3):
                    gen_image_[:, channel, :, :] = (gen_image_[:, channel, :, :] - self.norm_param[1, channel, 0]) * self.metrics / \
                                                   (self.norm_param[1, channel, 1] -
                                                    self.norm_param[1, channel, 0])
                D_real_decision = self.D(imgs, labels_)
                real_ = torch.ones_like(D_real_decision).to(device=self.device) * self.ratio_one
                D_real_loss = self.BCE_loss(D_real_decision, real_)

                D_fake_decision = self.D(imgs_, gen_image_)
                fake_ = (torch.zeros_like(D_fake_decision)+ self.ratio_zero).to(device=self.device)
                D_fake_loss = self.BCE_loss(D_fake_decision, fake_)

                # Back propagation
                D_loss = (D_real_loss + D_fake_loss) * 0.5 * self.ratio_dis


                # validate generator
                D_fake_decision = self.D(imgs_, gen_image)
                G_fake_loss = self.BCE_loss(D_fake_decision, real_)

                # L1 loss
                l1_loss_cholec_80 = self.ratio_cholec * self.L1_loss(gen_image_[:, :3, :, :], labels_[:, :3, :, :])
                l1_loss_depth = self.ratio_depth * self.L1_loss(gen_image_[:, 3:4, :, :], labels_[:, 3:4, :, :])
                cross_loss_seg = self.ratio_seg * self.Cross_loss(gen_seg[:, :, :, :], labels_[:, 4, :, :].\
                                                       to(torch.long))
                l1_loss = l1_loss_cholec_80 + l1_loss_depth + cross_loss_seg
                self.writer.add_scalars('monitor valid loss', {'loss of cholec80': l1_loss_cholec_80,
                                                               'loss of depth': l1_loss_depth,\
                                                               'loss of seg': cross_loss_seg}, cur_epoch)

                # Back propagation
                G_loss = G_fake_loss * self.ratio_dis + l1_loss
                D_losses.append(D_loss.item())

                G_losses.append(l1_loss.item())


            G_avg_loss = torch.mean(torch.Tensor(G_losses))
            D_avg_loss = torch.mean(torch.Tensor(D_losses))

            # avg loss values for plot
            self.G_valid_avg_losses.append(G_avg_loss)
            self.D_valid_avg_losses.append(D_avg_loss)
            print('{} Epoch {}, Generator valid loss {}, Discriminator valid loss {}'.format(
                datetime.datetime.now(), cur_epoch,
                G_avg_loss, D_avg_loss))
            fig, axes = plt.subplots(1, 7, figsize=(5, 5))
            imgs = [imgs[-1], gen_image[-1][:3, :, :], labels[-1][:3, :, :], \
                    gen_image[-1][3:4, :, :], labels[-1][3:4, :, :],\
                    gen_image[-1][4:, :, :], labels[-1][4:, :, :],]
            imgs_key = ['input', 'generate_cholec80', 'cholec80', \
                        "generate_depth", 'depth',\
                        "generate_seg", "seg"]
            show_imgs = []
            for key, ax, img in zip(imgs_key, axes.flatten(), imgs):
                ax.axis('off')
                ax.set_adjustable('box')
                # Scale to 0-255

                if key == "generate_depth":
                    img[0, :, :] = img[0, :, :] * \
                                         (self.norm_param[1, 3, 1] - self.norm_param[1, 3, 0]) \
                                         + self.norm_param[1, 3, 0]
                if key in ['input', 'generate_cholec80', 'cholec80',"generate_depth", 'depth']:
                    img = (((img - img.min()) * 255) / (img.max() - img.min())).detach().cpu() \
                        .numpy().transpose(1, 2, 0).astype(np.uint8)
                if key in ["generate_seg", "seg"]:
                    n2l_f = lambda x: self.seg_n2l[x]
                    img = img.detach().cpu().to(torch.long).apply_(n2l_f)\
                        .numpy().transpose(1, 2, 0).astype(np.uint8)

                show_imgs.append(img)
                ax.imshow(img, cmap=None, aspect='equal')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()
            self.writer.add_scalars('valid', {'Generator loss': G_avg_loss,
                                              'Discriminator loss': D_avg_loss}, cur_epoch)
            self.writer.add_image('valid_inpute image', show_imgs[0], cur_epoch, dataformats='HWC')
            self.writer.add_image('valid_cholec80 image', show_imgs[2], cur_epoch, dataformats='HWC')
            self.writer.add_image('valid_generated cholec80 image', show_imgs[1], cur_epoch, dataformats='HWC')
            self.writer.add_image('valid_depth image', show_imgs[4], cur_epoch, dataformats='HWC')
            self.writer.add_image('valid_generated depth image', show_imgs[3], cur_epoch, dataformats='HWC')
            self.writer.add_image('valid_seg image', show_imgs[6], cur_epoch, dataformats='HWC')
            self.writer.add_image('valid_generate seg image', show_imgs[5], cur_epoch, dataformats='HWC')

    def test(self, save_path='../../result/generated test images'):
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
        D_losses = []
        self.test_dataset = Images_Cholec_Depth_Seg_Test(self.data_Info, self.seg_l2n)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        monitor_path = save_path.replace('generated test images', 'monitor')
        if os.path.exists(monitor_path):
            shutil.rmtree(monitor_path)
        os.mkdir(monitor_path)
        with torch.no_grad():
            for cur_num, (ori_path, img, target, target_ori) in enumerate(self.test_dataset):
                splitted_ori_path = ori_path.split('\\')[1:]
                head_path = save_path
                for item in splitted_ori_path:
                    head_path = os.path.join(head_path, item)
                cholec80_path = head_path.replace('inputs', 'cholec80_style_02')
                depth_path = head_path.replace('inputs', 'depths').replace("img", "depth") \
                    .replace("png", "exr")
                seg_path = head_path.replace('inputs', 'labels').replace("img", "lbl")

                img = img.to(device=self.device)
                target = target.to(device=self.device)
                img_ = img.clone().unsqueeze(0)
                target_ = target.clone().unsqueeze(0)
                for channel in range(3):
                    img_[:, channel, :, :] = (img_[:, channel, :, :] - self.norm_param[0, channel, 0]) / \
                                             (self.norm_param[0, channel, 1] - self.norm_param[0, channel, 0])
                    target_[:, channel, :, :] = (target_[:, channel, :, :] - self.norm_param[1, channel, 0]) / \
                                                (self.norm_param[1, channel, 1] -
                                                 self.norm_param[1, channel, 0])
                channel = 3
                target_[:, channel, :, :] = (target_[:, channel, :, :] - self.norm_param[1, channel, 0]) / \
                                            (self.norm_param[1, channel, 1] -
                                             self.norm_param[1, channel, 0])
                gen_cholec80, gen_depth, gen_seg = self.G(img_)
                gen_cholec80_ = transforms.Resize(self.ori_size)(gen_cholec80).squeeze(0)
                gen_depth_ = transforms.Resize(self.ori_size)(gen_depth).squeeze(0)
                gen_seg_ = torch.argmax(gen_seg, dim=1, keepdim=True)
                gen_seg__ = transforms.Resize(self.ori_size, \
                                              interpolation=torchvision.transforms.InterpolationMode.NEAREST) \
                    (gen_seg_).squeeze(0)
                gen_image = torch.cat((gen_cholec80, gen_depth, gen_seg_), axis=1)
                D_real_decision = self.D(img_, target_)
                real_ = torch.ones_like(D_real_decision).to(device=self.device) * 0.99
                D_real_loss = self.BCE_loss(D_real_decision, real_)

                D_fake_decision = self.D(img_, gen_image)
                fake_ = torch.zeros_like(D_fake_decision).to(device=self.device)
                D_fake_loss = self.BCE_loss(D_fake_decision, fake_)

                # Back propagation
                D_loss = (D_real_loss + D_fake_loss) * 0.5 * 50

                # validate generator
                D_fake_decision = self.D(img_, gen_image)
                G_fake_loss = self.BCE_loss(D_fake_decision, real_)

                # L1 loss
                l1_loss_cholec_80 = 500 * self.L1_loss(gen_image[:, :3, :, :], target_[:, :3, :, :])
                l1_loss_depth = 250 * self.L1_loss(gen_image[:, 3:4, :, :], target_[:, 3:4, :, :])
                cross_loss_seg = 1 * self.Cross_loss(gen_seg[:, :, :, :], target_[:, 4, :, :]. \
                                                     to(torch.long))
                l1_loss = l1_loss_cholec_80 + l1_loss_depth + cross_loss_seg

                G_losses.append(l1_loss.item())
                D_losses.append(D_loss.item())
                file_name = ori_path.split('\\')[-1]
                self.writer.add_text("test", ' {}, L1 loss {}, Dis loss {}'.format( \
                    file_name, l1_loss.item(), D_loss.item()))
                print(' {}, L1 loss {}, Dis losss {}'.format(file_name, l1_loss.item(), \
                                                             D_loss.item()))
                # Scale to 0-255
                fig, axes = plt.subplots(1, 7, figsize=(5, 5))
                imgs = [target_ori[:3, :, :], gen_cholec80_, target_ori[3:6, :, :], \
                        gen_depth_, target_ori[6:7, :, :], \
                        gen_seg__, target_ori[7:8, :, :], ]
                imgs_key = ['input', 'generate_cholec80', 'cholec80', \
                            "generate_depth", 'depth', \
                            "generate_seg", "seg"]
                show_imgs = []
                for key, ax, img_temp in zip(imgs_key, axes.flatten(), imgs):
                    ax.axis('off')
                    ax.set_adjustable('box')
                    # Scale to 0-255
                    if key == "generate_cholec80":
                        for channel in range(3):
                            img_temp[channel, :, :] = img_temp[channel, :, :] * \
                                                      (self.norm_param[1, channel, 1] - self.norm_param[1, channel, 0]) \
                                                      + self.norm_param[1, channel, 0]
                    if key == "generate_depth":
                        img_temp[0, :, :] = img_temp[0, :, :] * \
                                            (self.norm_param[1, 3, 1] - self.norm_param[1, 3, 0]) \
                                            + self.norm_param[1, 3, 0]
                    if key in ["input", 'generate_cholec80', 'cholec80', "generate_depth", 'depth']:
                        img_temp = (((img_temp - img_temp.min()) * 255) / (
                                img_temp.max() - img_temp.min())).detach().cpu() \
                            .numpy().transpose(1, 2, 0).astype(np.uint8)
                    if key in ["generate_seg", "seg"]:
                        n2l_f = lambda x: self.seg_n2l[x]
                        img_temp = img_temp.detach().cpu().to(torch.long).apply_(n2l_f) \
                            .numpy().transpose(1, 2, 0).astype(np.uint8)

                    show_imgs.append(img_temp)
                    ax.imshow(img_temp, cmap=None, aspect='equal')
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.show()
                self.writer.add_scalars('test', {'Generator loss': G_losses[-1],
                                                 'Discriminator loss': D_losses[-1]}, cur_num)
                self.writer.add_image('test_inpute image', show_imgs[0], cur_num, dataformats='HWC')
                self.writer.add_image('test_cholec80 image', show_imgs[2], cur_num, dataformats='HWC')
                self.writer.add_image('test_generated cholec80 image', show_imgs[1], cur_num, dataformats='HWC')
                self.writer.add_image('test_depth image', show_imgs[4], cur_num, dataformats='HWC')
                self.writer.add_image('test_generated depth image', show_imgs[3], cur_num, dataformats='HWC')
                self.writer.add_image('test_seg image', show_imgs[6], cur_num, dataformats='HWC')
                self.writer.add_image('test_generate seg image', show_imgs[5], cur_num, dataformats='HWC')

                for path in [cholec80_path, depth_path, seg_path]:
                    filepath, _ = os.path.split(path)
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                cholec_img = show_imgs[1]
                cholec_ori = show_imgs[2]
                ori_img = show_imgs[0]
                Image.fromarray(show_imgs[1]).save(cholec80_path, "PNG")
                depth_img_ = np.concatenate((show_imgs[3], show_imgs[3], \
                                             show_imgs[3], show_imgs[3]), axis=2)
                depth_img = np.concatenate((show_imgs[3], show_imgs[3], \
                                            show_imgs[3]), axis=2)
                depth_ori = np.concatenate((show_imgs[4], show_imgs[4], \
                                            show_imgs[4]), axis=2)
                seg_img = np.concatenate((show_imgs[5], show_imgs[5], show_imgs[5]), axis=2)
                seg_ori = np.concatenate((show_imgs[6], show_imgs[6], show_imgs[6]), axis=2)
                Image.fromarray(seg_img).save(seg_path, "PNG")
                depth_img_ = depth_img_.astype(np.float32)
                cv2.imwrite(depth_path, depth_img_)
                file_name = ori_path.split('\\')[-1]
                folder_name = ori_path.split('\\')[1]
                monitor_path_cur = os.path.join(monitor_path, folder_name, file_name)
                filepath_monitor, _ = os.path.split(monitor_path_cur)
                if not os.path.exists(filepath_monitor):
                    os.makedirs(filepath_monitor)
                monitor_img_1 = np.concatenate((ori_img, cholec_ori, depth_ori, seg_ori), axis=1)
                monitor_img_2 = np.concatenate((ori_img, cholec_img, depth_img, seg_img), axis=1)
                monitor_img = np.concatenate((monitor_img_1, monitor_img_2), axis=0)
                Image.fromarray(monitor_img).save(monitor_path_cur, "PNG")
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
        self.D.load_state_dict(cur_checkpoints['discriminator'])
        self.G_optimizer.load_state_dict(cur_checkpoints['G_optimizer'])
        self.D_optimizer.load_state_dict(cur_checkpoints['D_optimizer'])
        self.D_avg_losses = []
        self.G_avg_losses = []
        self.G_valid_avg_losses = []

    def save(self,cur_epoch):
        self.check_points.save(cur_epoch, self.G, self.D, self.G_optimizer, self.D_optimizer)