"""
put cholec80 and depth channel together
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "True"
import torchdatasets as td
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import random
import torch
import cv2
import numpy as np

class Images(td.Dataset):  # Different inheritance
   def __init__(self, input_list):
       super().__init__()  # This is the only change
       self.input_list = input_list

   def __getitem__(self, index):
       return Image.open(self.input_list[index])

   def __len__(self):
       return len(self.input_list)

def Image_Aug(data_Info, key='input'):
   """
   parameter: dictionary(for train/valid/test)
   output: dataset(which consists of original, rotation(90,180), flipp(hor,ver))
   """

   original = Images(data_Info[key]).map(torchvision.transforms.ToTensor()) \
       .map(torchvision.transforms.Resize((256, 256))) \


   rotation_90 = original.map(transforms.RandomRotation((90, 90)))#.cache()
   rotation_180 = original.map(transforms.RandomRotation((180, 180)))#.cache()
   flip_hor = original.map(transforms.RandomHorizontalFlip(p=1))#.cache()
   flip_ver = original.map(transforms.RandomVerticalFlip(p=1))#.cache()
   datasets = [original, rotation_90, rotation_180, flip_hor, flip_ver]
   return datasets

def Image_Ori(data_Info, key='input'):
   """
   parameter: dictionary(for train/valid/test)
   output: dataset(which consists of original, rotation(90,180), flipp(hor,ver))
   """

   original = Images(data_Info[key] \
                     ).map(torchvision.transforms.ToTensor()) \
       .map(torchvision.transforms.Resize((256, 256))) \
       .cache()
   return [original]

class Images_Data_Test(Dataset):
    def __init__(self, input_list):
        self.data_Info = input_list
        self.input = self.data_Info['input']
        self.target = self.data_Info['cholec80']

    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        path = self.input[idx]
        img_input = Image.open(path)
        img_target = Image.open(self.target[idx])
        transformation = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        return path, transformation(img_input), transforms.ToTensor()(img_target)



class Images_Dataset(Dataset):
   def __init__(self, data_Info, key_type):
       self.data_Info = data_Info
       self.key_type = key_type
       if key_type == 'train':
           self.input = Image_Aug(data_Info=self.data_Info[self.key_type], \
                                  key='input')
           self.target = Image_Aug(data_Info=self.data_Info[self.key_type], \
                                   key='cholec80')
       else:
           self.input = Image_Ori(data_Info=self.data_Info[self.key_type], \
                                  key='input')
           self.target = Image_Ori(data_Info=self.data_Info[self.key_type], \
                                   key='cholec80')


   def __len__(self):

       return len(self.input[0])

   def __getitem__(self, idx):

       tf_idx = random.choice(range(len(self.input)))
       return self.input[tf_idx][idx], self.target[tf_idx][idx]


def get_normalize_parameter(data_Info):
    """

    :param data_Info: the opened json file, the dictionary which includes the information for train/test/valid
    :return: the mean and var value of input dataset, which is used for normalize
    """
    norm_param = dict()
    for norm_key in ['input','cholec80']:
        paths = data_Info['train'][norm_key]
        num_sample = len(paths)
        total_sample = torch.zeros(num_sample,3,256,256)
        transformation = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        for i in range(num_sample):
            img = Image.open(paths[i])
            total_sample[i] = transformation(img)
        mean = []
        var = []
        for i in range(3):
            mean.append(total_sample[:,i,:,:].min().item())
            var.append(total_sample[:,i,:,:].max().item())
        key_dict = dict()
        key_dict['min'] = mean
        key_dict['max'] = var
        norm_param[norm_key] = key_dict
    # calculate min and max of depth channel
    norm_key = 'depth'
    paths = data_Info['train'][norm_key]
    num_sample = len(paths)
    total_sample = torch.zeros(num_sample,1,256,256)
    transformation = transforms.Compose([transforms.Resize((256, 256))])
    for i in range(num_sample):
        img = cv2.imread(paths[i], cv2.IMREAD_UNCHANGED)[:,:,0:1]
        img = torch.tensor(img).permute(2,0,1)
        total_sample[i] = transformation(img)
    mean = []
    var = []
    mean.append(total_sample[:,0,:,:].min().item())
    var.append(total_sample[:,0,:,:].max().item())
    key_dict = dict()
    key_dict['min'] = mean
    key_dict['max'] = var
    norm_param[norm_key] = key_dict
    ret_tensor = torch.zeros(2,4,2)
    for i in range(3):
        ret_tensor[0,i,0] = norm_param['input']['min'][i]
        ret_tensor[0, i, 1] = norm_param['input']['max'][i]
        ret_tensor[1, i, 0] = norm_param['cholec80']['min'][i]
        ret_tensor[1, i, 1] = norm_param['cholec80']['max'][i]
    ret_tensor[1, 3, 0] = norm_param['depth']['min'][0]
    ret_tensor[1, 3, 1] = norm_param['depth']['max'][0]

    #  generate dictionary map for seg channel
    initial_s = set([])
    for norm_key in['train','test','valid']:
        paths = data_Info[norm_key]['seg']
        num_sample = len(paths)

        for i in range(num_sample):
            temp = Image.open(paths[i])
            initial_s = initial_s | set(list(np.unique(np.array(temp))))
    seg_list = list(initial_s)
    seg_list.sort()

    ret_l2n_dict = {}
    ret_n2l_dict = {}
    for i in range(len(seg_list)):
        ret_l2n_dict[seg_list[i]] = i
        ret_n2l_dict[i] = seg_list[i]
    print(ret_l2n_dict)
    print(ret_n2l_dict)

    return ret_tensor, ret_l2n_dict, ret_n2l_dict


class Depths(td.Dataset):  # Different inheritance
   def __init__(self, input_list):
       super().__init__()  # This is the only change
       self.input_list = input_list

   def __getitem__(self, index):
    return torch.tensor(cv2.imread(self.input_list[index], cv2.IMREAD_UNCHANGED)[:,:,0:1]).permute(2,0,1)
   def __len__(self):
       return len(self.input_list)

def Depth_Aug(data_Info, key='depth'):
   """
   parameter: dictionary(for train/valid/test)
   output: dataset(which consists of original, rotation(90,180), flipp(hor,ver))
   """

   original = Depths(data_Info[key]).map(torchvision.transforms.Resize((256, 256))) \
       #.cache()
   rotation_90 = original.map(transforms.RandomRotation((90, 90)))#.cache()
   rotation_180 = original.map(transforms.RandomRotation((180, 180)))#.cache()
   flip_hor = original.map(transforms.RandomHorizontalFlip(p=1))#.cache()
   flip_ver = original.map(transforms.RandomVerticalFlip(p=1))#.cache()
   datasets = [original, rotation_90, rotation_180, flip_hor, flip_ver]
   return datasets
def Depth_Ori(data_Info, key='depth'):
   """
   parameter: dictionary(for train/valid/test)
   output: dataset(which consists of original)
   """

   original = Depths(data_Info[key])\
       .map(torchvision.transforms.Resize((256, 256))) \
       .cache()
   return [original]

class Depths_Data_Test(Dataset):
    def __init__(self, input_list):
        self.data_Info = input_list
        self.input = self.data_Info['input']
        self.target = self.data_Info['depth']

    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        path = self.input[idx]
        img_input = Image.open(path)
        img_target = cv2.imread(self.target[idx], cv2.IMREAD_UNCHANGED)[:,:,0:1]
        transformation = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        return path, transformation(img_input), transforms.ToTensor()(img_target)



class Depths_Dataset(Dataset):
   def __init__(self, data_Info, key_type='train'):
       self.data_Info = data_Info
       self.key_type = key_type

       if key_type == 'train':
           self.input = Image_Aug(data_Info=self.data_Info[self.key_type], \
                                  key='input')
           self.target = Depth_Aug(data_Info=self.data_Info[self.key_type], \
                                   key='depth')
       else:
           self.input = Image_Ori(data_Info=self.data_Info[self.key_type], \
                                  key='input')
           self.target = Depth_Ori(data_Info=self.data_Info[self.key_type], \
                                   key='depth')


   def __len__(self):

       return len(self.input[0])

   def __getitem__(self, idx):

       tf_idx = random.choice(range(len(self.input)))
       return self.input[tf_idx][idx], self.target[tf_idx][idx]





class Segs(td.Dataset):  # Different inheritance
   def __init__(self, input_list):
       super().__init__()  # This is the only change
       self.input_list = input_list

   def __getitem__(self, index):
    return torch.tensor(np.array(Image.open(self.input_list[index]))[:,:,0:1]).permute(2,0,1)
   def __len__(self):
       return len(self.input_list)


def Seg_Aug(data_Info, key='seg'):
   """
   parameter: dictionary(for train/valid/test)
   output: dataset(which consists of original, rotation(90,180), flipp(hor,ver))
   """

   original = Segs(data_Info[key]).map(torchvision.transforms.Resize((256, 256),interpolation=\
                                                                     torchvision.transforms.InterpolationMode.NEAREST)) \
       #.cache()
   rotation_90 = original.map(transforms.RandomRotation((90, 90)))#.cache()
   rotation_180 = original.map(transforms.RandomRotation((180, 180)))#.cache()
   flip_hor = original.map(transforms.RandomHorizontalFlip(p=1))#.cache()
   flip_ver = original.map(transforms.RandomVerticalFlip(p=1))#.cache()
   datasets = [original, rotation_90, rotation_180, flip_hor, flip_ver]
   return datasets
def Seg_Ori(data_Info, key='seg'):
   """
   parameter: dictionary(for train/valid/test)
   output: dataset(which consists of original)
   """

   original = Segs(data_Info[key]).map(torchvision.transforms.Resize((256, 256), interpolation= \
       torchvision.transforms.InterpolationMode.NEAREST)) \
       # .cache()
   return [original]

class Seg_Data_Test(Dataset):
    def __init__(self, input_list):
        self.data_Info = input_list
        self.input = self.data_Info['input']
        self.target = self.data_Info['seg']

    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        path = self.input[idx]
        img_input = Image.open(path)
        img_target = cv2.imread(self.target[idx], cv2.IMREAD_UNCHANGED)[:,:,0:1]
        transformation = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        return path, transformation(img_input), transforms.ToTensor()(img_target)


class Cholec_Depth_Seg_Dataset(Dataset):
   def __init__(self, data_Info, seg_l2n,key_type='train'):
       self.data_Info = data_Info
       self.key_type = key_type
       self.seg_l2n = seg_l2n
       if key_type == 'train':
           self.input = Image_Aug(data_Info=self.data_Info[self.key_type], \
                                  key='input')
           self.cholec = Image_Aug(data_Info=self.data_Info[self.key_type], \
                                   key='cholec80')
           self.depth = Depth_Aug(data_Info=self.data_Info[self.key_type], \
                                   key='depth')
           self.seg = Seg_Aug(data_Info=self.data_Info[self.key_type], \
                                   key='seg')

       else:
           self.input = Image_Ori(data_Info=self.data_Info[self.key_type], \
                                  key='input')
           self.cholec = Image_Ori(data_Info=self.data_Info[self.key_type], \
                                   key='cholec80')
           self.depth = Depth_Ori(data_Info=self.data_Info[self.key_type], \
                                  key='depth')
           self.seg = Seg_Ori(data_Info=self.data_Info[self.key_type], \
                                   key='seg')

   def __len__(self):

       return len(self.input[0])

   def __getitem__(self, idx):

       tf_idx = random.choice(range(len(self.input)))
       ret_seg = self.seg[tf_idx][idx][:1,:,:].apply_(lambda x: \
                                                       self.seg_l2n[int(x)])
       return self.input[tf_idx][idx], torch.cat([self.cholec[tf_idx][idx], self.depth[tf_idx][idx],\
                                                  ret_seg], dim=0)

class Images_Cholec_Depth_Seg_Test(Dataset):
    def __init__(self, data_Info, seg_l2n, key_type='test'):
        self.data_Info = data_Info
        self.key_type = key_type
        self.seg_l2n = seg_l2n

        self.input = Image_Ori(data_Info=self.data_Info[self.key_type], \
                               key='input')
        self.cholec = Image_Ori(data_Info=self.data_Info[self.key_type], \
                                key='cholec80')
        self.depth = Depth_Ori(data_Info=self.data_Info[self.key_type], \
                               key='depth')
        self.seg = Seg_Ori(data_Info=self.data_Info[self.key_type], \
                           key='seg')

    def __len__(self):

        return len(self.input[0])

    def __getitem__(self, idx):
        path_input= self.data_Info[self.key_type]['input'][idx]
        path_cholec = self.data_Info[self.key_type]['cholec80'][idx]
        path_depth = self.data_Info[self.key_type]['depth'][idx]
        path_seg = self.data_Info[self.key_type]['seg'][idx]
        tf_idx = 0
        ret_seg = self.seg[tf_idx][idx][:1, :, :].apply_(lambda x: \
                                                             self.seg_l2n[int(x)])
        input_ori = transforms.ToTensor()(Image.open(path_input))
        cholec_ori = transforms.ToTensor()(Image.open(path_cholec))
        depth_ori = torch.tensor(cv2.imread(path_depth, cv2.IMREAD_UNCHANGED)[:,:,0:1]).permute(2,0,1)
        seg_temp = torch.tensor(np.array(Image.open(path_seg))[:,:,0:1]).permute(2,0,1)
        seg_ori = seg_temp.apply_(lambda x: self.seg_l2n[int(x)])
        target_ori = torch.cat([input_ori, cholec_ori, \
                                depth_ori, seg_ori], dim=0)
        target = torch.cat([self.cholec[tf_idx][idx], self.depth[tf_idx][idx], \
                                                   ret_seg], dim=0)
        return path_input,self.input[tf_idx][idx], target, target_ori
