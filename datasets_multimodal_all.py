#---------------------------------------------------------------------------------------------------#
# File name: datasets.py                                                                            #
# Autor: Chrissi2802                                                                                #
# Created on: 14.07.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.
# This file provides the dataset.
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class WISDM_Dataset():
    """Class to design a WISDM Dataset."""
    # def __init__(self, mode):
    #     """Initialisation of the class (constructor). It prepares the data to be used for training and validation."""
    #     # Input:
    #     # mode; string, train or test data
    #
    #     # Load the data
    #     self.column_names = ["BVP", "Timestamp", "Label"]
    #     self.acitvity_names = ["1", "2", "3", "4", "5"]
    #     self.label_dic = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    #
    #     # Inverted dictionary for reconversion
    #     self.activity_dic_inv = {item: element for element, item in self.label_dic.items()}
    #
    #     self.folder = "./Datasets-Test/" + mode + "-EYE(50Hz)" + "/"
    #     self.folder = "./Datasets-Test/" + mode + "-BVP(50Hz)" + "/"
    #     self.filelist = []
    #     # self.filelist = [csv for csv in os.listdir(self.folder) if csv[-4:] == ".csv"]
    #     # Get the list of subdirectories within the "mode" directory
    #     subdirs = [dir for dir in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, dir))]
    #     for subdir in subdirs:
    #         # Get the list of csv files in the current subdir
    #         for csv in os.listdir(os.path.join(self.folder, subdir)):
    #             if csv.endswith(".csv"):
    #                 self.filelist.append(os.path.join(subdir, csv))
    #
    #     self.data = pd.DataFrame()
    #
    #     self.__create_tensor()
    #
    #     self.predfname = None

    def __init__(self, mode, modalities):
        """Initialisation of the class (constructor). It prepares the data to be used for training and validation."""
        # Input:
        # modes: list of strings, each element is the name of a modality (e.g., ["EYE", "BVP", "EDA"])

        # Load the data
        self.column_names = ["BVP", "Timestamp", "Label"]
        self.acitvity_names = ["1", "2", "3", "4", "5"]
        self.label_dic = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

        # Inverted dictionary for reconversion
        self.activity_dic_inv = {item: element for element, item in self.label_dic.items()}

        self.modalities = modalities
        self.data = {}  # A dictionary to store different DataFrames of data and timestamps for each modality
        self.labels = {}  # A dictionary to store different DataFrames of labels for each modality
        self.timestamps = {}
        
        for modality in self.modalities:
            self.folder = "./Datasets-Test/" + mode + "-" + modality + "(50Hz)/"
            self.filelist = []
            # Get the list of subdirectories within the "mode" directory
            subdirs = [dir for dir in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, dir))]
            for subdir in subdirs:
                # Get the list of csv files in the current subdir
                for csv in os.listdir(os.path.join(self.folder, subdir)):
                    if csv.endswith(".csv"):
                        self.filelist.append(os.path.join(subdir, csv))

            self.data[modality] = pd.DataFrame()
            self.labels[modality] = pd.DataFrame()
            self.timestamps[modality] = pd.DataFrame()

            self.__create_tensor(modality)
            print("这里打印不同模态的self.data格式",modality,self.data[modality].shape,
                  modality,self.labels[modality].shape,self.timestamps[modality].shape)

        self.predfname = None


    def __create_tensor(self, modality):
        """This method combines all text files into one big tensor."""

        for csv in self.filelist:
            self.one_data = pd.read_csv(self.folder + csv, comment = ";") # load data

            # self.__normalize_feature()  # normalizes all features

            # 过滤标签为0的数据行
            self.one_data = self.one_data[self.one_data['Label'] != 0]
            print(modality,":",len(self.one_data))
            # save the normalized data in a tensor
            self.data[modality] = pd.concat([self.data[modality], self.one_data.iloc[:, :-2]], ignore_index=True)
            self.labels[modality] = pd.concat([self.labels[modality], self.one_data[['Label']]], ignore_index=True)
            self.timestamps[modality] = pd.concat([self.timestamps[modality], self.one_data[['Timestamp']]], ignore_index=True)

        self.__normalize_feature_all(modality)


    def __normalize_feature(self):
        """This method normalizes all features."""

        for dim in self.one_data.columns[:-2]:
            # normalize the data
            mue = self.one_data[dim].mean()  # Mean
            sigma = self.one_data[dim].std()  # Standard deviation
            self.one_data[dim] = (self.one_data[dim] - mue) / sigma

    def __normalize_feature_all(self, modality):
        """This method normalizes all features."""

        for dim in self.data[modality].columns[:-2]:
            # normalize the data
            mue = self.data[modality][dim].mean()  # Mean
            sigma = self.data[modality][dim].std()  # Standard deviation
            self.data[modality][dim] = (self.data[modality][dim] - mue) / sigma


    def dataloading(self, batch_size, shuffle, drop_last, sliding_window = False):
        """This method fills the DataLoader."""
        # Input:
        # batch_size; integer, batch size
        # shuffle; boolean, shuffle the data in the DataLoader
        # drop_last; boolean, delete last batch (Sometimes incomplete)
        # sliding_window; boolean

        if (sliding_window == True):
            self.data_tensor = self.slid_win(self.data_tensor, 100, 50)

        # if (sliding_window == True):
        #     self.data_tensor = self.slid_win(self.data_tensor, 128, 64)      #测试用

        # Data loading
        dl = DataLoader(self.data_tensor, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)

        return dl

    def writepredictions(self, sample_id, prediction, model_name):
        """This method evaluates the passed predictions and writes a new line into a text file for each sample_id."""
        # Input:
        # sample_id; integer, current sample_id
        # prediction; torch tensor, Contains the predicted labels for a sample_id
        # model_name; string, name of the ANN model

        # Create a unique name for the text file
        if (self.predfname == None):
            self.predfname = model_name + "_Predictions_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
            
            # Save the header
            file = open("./Predictions/" + self.predfname, "a")
            file.write("sample_id,activity\n")
            file.close()
        
        # Determine the activity that was predicted the most 
        # And convert this number back to the corresponding letter
        activity = prediction.mode(dim = 0)[0].item()
        activity = self.activity_dic_inv.get(activity)

        # Save these two values in a text file (sample_id,activity)
        file = open("./Predictions/" + self.predfname, "a")
        file.write(str(sample_id) + "," + activity + "\n")
        file.close()

    def visualisation(self):
        """This method visualises the data."""

        self.__vis_data_points_per_category()   # Number of data points in each category as bar chart
        # self.__vis_sample_series_per_category() # Sample data series for all six categories
        
    def __vis_data_points_per_category(self):
        """This method displays the number of data points in each category as a bar chart."""

        activity_counts = torch.unique(self.data[:, -1].long(), sorted = True, return_counts = True)
        mean = torch.mean(activity_counts[1].float())
        total_count = torch.sum(activity_counts[1].float())  # 计算总计数
        
        plt.rcParams["figure.figsize"] = (12, 7)
        plt.bar(self.acitvity_names, activity_counts[1], label = "Number of data points")
        plt.axhline(mean, label = "Mean", color = "red")
        plt.title("Number of datapoints by Activities")
        plt.legend()
        # 添加文本标注
        # for i in range(len(activity_counts[1])):
        #     plt.text(i, activity_counts[1][i].item(), str(activity_counts[1][i].item()), ha='center', va='bottom')
        # 添加文本标注
        for i in range(len(activity_counts[1])):
            count = activity_counts[1][i].item()
            percentage = (count / total_count) * 100  # 计算百分比
            plt.text(i, count, f"{count} ({percentage:.2f}%)", ha='center', va='bottom')
        plt.savefig("Number_of_datapoints_by_Activities.png")
        plt.show()

    def __vis_sample_series_per_category(self):
        """This method visualises sample data series for all six categories."""
        
        length = 200
        labels = ["x-signal", "y-signal", "z-signal"]
        x_values = np.linspace(0.0, length * 0.05, length)

        fig, axes = plt.subplots(3, 2, sharex = True, figsize = (18, 9))

        for i in range(6):  # Fill all subplots
            start = i * 2400
            tensorxyz = self.data_tensor_raw[start:start + length, 2:5]   # data for this plot

            if (i < 3):
                row = i
                col = 0
            else:
                row = i - 3
                col = 1
                
            axes[row, col].plot(x_values, tensorxyz)
            axes[row, col].set_title(self.acitvity_names[i])
            axes[row, col].grid()

        fig.legend(labels)
        plt.setp(axes[-1, :], xlabel = "Time [s]")
        plt.suptitle("Sample data series of each category")
        plt.savefig("Sample_data_series_of_each_category.png")
        plt.show()
    
    def slid_win(self, data, window_size, step_size):
        """This method implements a sliding window."""
        # Input:
        # window_size; integer
        # step_size; integer
        # Output:

        output = []

        for i in range(0, data.shape[0] - window_size, step_size):
            
            local_data = data[i: i + window_size]
            local_data = torch.unsqueeze(local_data, 0)
            #print(local_data.shape)

            if (output != []):
                output = torch.cat((output, local_data))
            else:
                output = local_data

        return output


class Create_Dataset(Dataset):
    """Class to design a Dataset."""

    def __init__(self, X, Y, T, time_length, sliding_step):
        """Initialisation of the class (constructor)."""
        # Input:
        # X
        # Y
        # T
        # time_length
        # sliding_step
        
        super().__init__()
        
        data = []
        labels = []
        frequency = 1000 / (T.values[1] - T.values[0])

        i = 0  # 初始化i为0

        while i < len(X) - time_length + 1:
            if (Y.values[i] == Y.values[i + time_length - 1] and T.values[i + time_length - 1] - T.values[i] <= (time_length / frequency) * 1000):
                data.append(torch.from_numpy(X.values[i: i + time_length].astype(np.float)).float())
                if ('Label' in Y):
                    labels.append(Y['Label'].values[i])
                else:
                    labels.append(Y['test-id'].values[i])
                i += sliding_step  # 满足条件时，以sliding_step为单位增加i
            else:
                i += 1
                # 寻找下一个可滑动窗口的起始点
                while i < len(X) - time_length + 1:
                    if (Y.values[i] != Y.values[i - 1] or T.values[i] - T.values[i - 1] > (1 / frequency) * 1000):
                        break
                    i += 1

        self.data = torch.stack(data) # Shape = [num_samples, time_length, features=3]
        self.labels = torch.tensor(labels) # Shape = [num_samples]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 创建一个数据预处理类，接收预先读取好的数据和标签，并根据时间戳分割数据
class MultiModalityDataset(Dataset):
    def __init__(self, X_dict, Y_dict, T_dict,time_length,sliding_step):
        self.X_dict = X_dict
        self.Y_dict = Y_dict
        self.T_dict = T_dict
        self.time_length = time_length
        self.sliding_step = sliding_step
        # self.T = pd.read_csv("./Datasets-Test/multimodal/Start Timestamps Of Segments.csv")
        #self.T = pd.read_csv("./Datasets-Test/multimodal/Start Timestamps Of Segments - Labeled.csv")
        self.dict = {}
        self.labels = {}
    #     self._split_data_by_timestamps()

    # def _split_data_by_timestamps(self):
    #     is_label = True
        #frequency = 1000 / (T.values[1] - T.values[0])

        for modality in self.X_dict.keys():
            data = []
            labels = []
            X = self.X_dict[modality]
            # X.set_index('Timestamp', inplace=True)
            Y = self.Y_dict[modality]
            T = self.T_dict[modality]
            # print(X.index[:5].values)

            i = 0  # 初始化i为0

            while i < len(X) - time_length + 1:
                if (Y.values[i] == Y.values[i + time_length - 1] and T.values[i + time_length - 1] - T.values[i] <= (
                        time_length / 50) * 1000):
                    data.append(torch.from_numpy(X.values[i: i + time_length].astype(float)).float())
                    if ('Label' in Y):
                        labels.append(Y['Label'].values[i])
                    else:
                        labels.append(Y['test-id'].values[i])
                    i += sliding_step  # 满足条件时，以sliding_step为单位增加i
                else:
                    i += 1
                    # 寻找下一个可滑动窗口的起始点
                    while i < len(X) - time_length + 1:
                        if (Y.values[i] != Y.values[i - 1] or T.values[i] - T.values[i - 1] > (1 / 50) * 1000):
                            break
                        i += 1

            # for timestamp in self.T:
            #     i_start = X.index.get_loc(timestamp)
            #     i_end = i_start + self.time_length
            #
            #     if i_end < len(X):
            #         data.append(torch.from_numpy(X.values[i_start: i_end].astype(np.float)).float())
            #         if is_label == True:
            #             labels.append(Y.values[i_start])
            self.dict[modality] = torch.stack(data)
            
            self.labels[modality]= torch.tensor(labels)

        # min_height = min(tensor.size(0) for tensor in self.dict.values())
        # min_width = min(tensor.size(1) for tensor in self.dict.values())

        # # 使用列表推导式将所有的三维张量堆叠在一起，并进行裁剪或填充
        # stacked_tensors = []
        # for tensor in self.dict.values():
        #     # 裁剪或填充
        #     if tensor.size(0) > min_height or tensor.size(1) > min_width:
        #         tensor = tensor[:min_height, :min_width, ...]  # 裁剪
        #     elif tensor.size(0) < min_height or tensor.size(1) < min_width:
        #         # 创建一个与目标尺寸相同的新张量，并用0填充
        #         padded_tensor = torch.zeros(min_height, min_width, tensor.size(2))
        #         padded_tensor[:tensor.size(0), :tensor.size(1), ...] = tensor  # 填充
        #         tensor = padded_tensor

        #     stacked_tensors.append(tensor)

        # # 在最后一维上拼接
        # self.data = torch.cat(stacked_tensors, dim=-1)

        # # 裁剪或填充标签张量
        # if self.labels.size(0) > min_height or self.labels.size(1) > min_width:
        #     self.labels = self.labels[:min_height]  # 裁剪
        # elif self.labels.size(0) < min_height or self.labels.size(1) < min_width:
        #     # 创建一个与目标尺寸相同的新张量，并用0填充
        #     padded_labels = torch.zeros(min_height)
        #     padded_labels[:self.labels.size(0)] = self.labels  # 填充
        #     self.labels = padded_labels
            


        stacked_tensors = [tensor for tensor in self.dict.values()]
        # 打印合并后的张量大小
        self.data = torch.cat(stacked_tensors, dim=-1)
        print(self.data.shape)
        

  

   
       

 


    def __len__(self):
        return len(self.labels)

    # def __getitem__(self, idx):
    #     return {'EYE': self.data_dict['EYE'][idx], 'ACC': self.data_dict['ACC'][idx],
    #             'BVP': self.data_dict['BVP'][idx], 'EDA': self.data_dict['EDA'][idx],
    #             'TEMP': self.data_dict['TEMP'][idx], 'HR': self.data_dict['HR'][idx],
    #             'label': self.labels[idx]}
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


