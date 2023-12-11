import torchdatasets as td
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import random
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
       #.cache()
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
        self.target = self.data_Info['target']

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
                                   key='target')
       else:
           self.input = Image_Ori(data_Info=self.data_Info[self.key_type], \
                                  key='input')
           self.target = Image_Ori(data_Info=self.data_Info[self.key_type], \
                                   key='target')


   def __len__(self):

       return len(self.input[0])

   def __getitem__(self, idx):

       tf_idx = random.choice(range(len(self.input)))
       return self.input[tf_idx][idx], self.target[tf_idx][idx]
