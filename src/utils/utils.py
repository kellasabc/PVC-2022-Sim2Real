import torch
import os
import shutil
import numpy as np
import glob
import json


class check_points():
    def __init__(self, folder="./result/model/train_valid_test"):
        """
        :param folder: the folder which save the model
        """
        self.folder = folder
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.makedirs(self.folder)
    def save(self, epoch, G, D, G_opt, D_opt):
        """
        save parameter of generator, discriminator, optimization of generator and optimization of discriminator as a
        dictionary in a .tar file
        :param epoch: current epoch
        :param G: generator
        :param D: discriminator
        :param G_opt: optimization of generator
        :param D_opt: optimization of discriminator
        :return: None
        """
        file_name = "epoch_{}.tar".format(epoch)
        path = os.path.join(self.folder, file_name)
        torch.save({
            'epoch': epoch,
            'generator': G.state_dict(),
            'discriminator': D.state_dict(),
            "G_optimizer": G_opt.state_dict(),
            "D_optimizer": D_opt.state_dict(),
        }, path)


    def load(self,file_name):
        """
        load the saved checkpoints
        :param file_name: the file name of saved mode, i.g."epoch_100.tar"
        :return:
        """
        path = os.path.join(self.folder, file_name)
        if os.path.exists(path):
            return torch.load(path)
        else:
            print("the given file_name:{} is wrong".format(file_name))
            return None

class data_json():
    def __init__(self, folder="./result/json files"):
        """
        initialize the folder, which is used to save generated json files
        :param folder: folder, which is used to save generated json files
        """
        self.folder = folder
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.makedirs(self.folder)

    def train_valid_test(self, train_radio=0.6, \
                       valid_radio=0.2, \
                       simulated_data_path='../../data/simulated', \
                       file_Name='train_valid_test.json', \
                       seed=1024):
        """
        according to the radio to save the path of train, valid, test dataset in a json file
        here to get the traget path the string replaced method are used according to the difference between\
        input and output
        the form of input_path: "../../data/simulated/*/input/*.png"
        the form of target_path: '../../data/styleFromCholec80/*/style../*.png'
        generated json file are saved as './result/json files/train_valid_test/train_valid_test.json'
        :param train_radio: the radio of train dataset
        :param valid_radio: the radio of train dataset
        :param simulated_data_path: the folder which save the input dataset
        :param file_Name: the name of generated json file
        :param seed: the seed for np.random.seed
        :return: None
        """

        output_folder = os.path.join(self.folder, "train_valid_test")
        np.random.seed(seed)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        train_path = glob.glob(simulated_data_path + '/*/*/*.png')
        np.random.shuffle(train_path)
        target_path = [i.replace("simulated", "styleFromCholec80") \
                           .replace("inputs", "style_02") \
                       for i in train_path]

        total_Num = len(train_path)
        train_Num, valid_Num = int(total_Num * train_radio), int(total_Num * valid_radio)
        test_Num = total_Num - train_Num - valid_Num
        ret_Dic = {}
        train_Dataset = {'input': train_path[: train_Num], 'target': target_path[: train_Num]}
        valid_Dataset = {'input': train_path[train_Num: train_Num + test_Num], \
                         'target': target_path[train_Num: train_Num + test_Num]}
        test_Dataset = {'input': train_path[train_Num + test_Num:], \
                        'target': target_path[train_Num + test_Num:]}
        ret_Dic['train'] = train_Dataset
        ret_Dic['valid'] = valid_Dataset
        ret_Dic['test'] = test_Dataset
        output_dict_pth = os.path.join(output_folder, file_Name)
        with open(output_dict_pth, 'w', encoding='utf-8') as f:
            json.dump(ret_Dic, f, indent=4)
            print("load data finished!")


    def k_folder(self,seed =1024,test_radio = 0.2, \
                 simulated_data_path='../../data/simulated',\
                 k = 5):
        """
        generate the json file for the k_folder method
        the form of input_path: "../../data/simulated/*/input/*.png"
        the form of target_path: '../../data/styleFromCholec80/*/style../*.png'
        generated json file are saved as './result/json files/k_*_folder/k_*.json'
        :param seed: the seed for np.random.seed
        :param test_radio: the radio of test dataset
        :param simulated_data_path:
        :param target_path:
        :param k: how many folder you want to generate
        :return: None
        """
        np.random.seed(seed)
        output_folder = os.path.join(self.folder, "k_{}_folder").format(k)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        train_path = glob.glob(simulated_data_path + '/*/*/*.png')
        np.random.shuffle(train_path)
        target_path = [i.replace("simulated", "styleFromCholec80") \
                           .replace("inputs", "style_02") \
                       for i in train_path]

        total_Num = len(train_path)
        split_total_Num = len(train_path) - int(total_Num * test_radio)
        split_Points = [int(i * split_total_Num / k) for i in range(k)]
        split_Points.append(split_total_Num)

        end_point = split_Points[-1]
        for i, split_Point in enumerate(split_Points[:-1]):
            file_Name = 'k_{}.json'.format(i+1)
            before, after = split_Point, split_Points[i + 1]
            k_train_I_path = train_path[0: before] + train_path[after: end_point]
            k_train_T_path = target_path[0: before] + target_path[after: end_point]
            k_valid_I_path = train_path[before: after]
            k_valid_T_path = target_path[before: after]
            train_Dataset = {'input': k_train_I_path, 'target': k_train_T_path}
            valid_Dataset = {'input': k_valid_I_path, 'target': k_valid_T_path}
            ret_Dic = {}
            ret_Dic['train'] = train_Dataset
            ret_Dic['valid'] = valid_Dataset
            output_dict_pth = os.path.join(output_folder, file_Name)
            with open(output_dict_pth, 'w', encoding='utf-8') as f:
                json.dump(ret_Dic, f, indent=4)
                print("load {}_folfer data finished...".format(i))
        file_Name = 'test.json'
        output_dict_pth = os.path.join(output_folder, file_Name)
        ret_Dic = {}
        test_Dataset = {'input': train_path[end_point:], \
                        'target': target_path[end_point:]}
        ret_Dic['test'] = test_Dataset
        with open(output_dict_pth, 'w', encoding='utf-8') as f:
            json.dump(ret_Dic, f, indent=4)
            print("load test data finished...")

