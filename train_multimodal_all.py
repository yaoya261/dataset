#---------------------------------------------------------------------------------------------------#
# File name: train.py                                                                               #
# Autor: Chrissi2802                                                                                #
# Created on: 14.08.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# WISDM - Biometric time series data classification
# Exact description in the functions.
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
import datasets_multimodal_all,models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 设置随机种子
# torch.manual_seed(42)


#------------------------------------------------------------------------------------------------------------------#
#                                                MLP & CNN 1D & GRU                                                #
#------------------------------------------------------------------------------------------------------------------#


def train_mlp_cnnv1_gru(model, epochs, batch_size, learning_rate, cuda, plots):
    """This function trains a model (MLP, CNN 1D, GRU) for classification using the WISDM dataset."""
    # Input: 
    # model; pytorch model
    # epochs; number of epochs
    # batch_size; number training batch size
    # learning_rate; number learning rate
    # cuda; boolean train the model on cuda or not
    # plots; boolean produce plots of train and test losses and accuracies
    # Output:
    # model; the pytorch trained model
    # train_losses; where train losses are a simple python list

    # Load the data and put it into the DataLoader
    print("Prepare the data for training ...")
    dataset_train = datasets_multimodal_all.WISDM_Dataset("train")

    if (model.__class__.__name__ == "GRU_NET"):
        sw = True   # use sliding_window
    else:
        sw = False

    dl_train = dataset_train.dataloading(batch_size, True, True, sliding_window = sw)
    print("Preparation of the data completed!")

    if ((plots == True) and (sw == False)):
        dataset_train.visualisation()

    train_losses = []
    train_acc = []

    if (cuda == True):
        # moving model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(device)   # only for testing
        model = model.to(device)
    else:
        device = "cpu"

    loss = nn.CrossEntropyLoss()    # Classification => Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # train Model
    print("Start of training ...")
    for epoch in range(epochs):
        
        # Training
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0

        for batch in dl_train:
            x_batch, y_batch = batch[:, 2:5], batch[:, 1].long()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()   # reset gradients to avoid incorrect calculation
            prediction = model.forward(x_batch)
            l = loss(prediction, y_batch)
            l.backward()
            optimizer.step()
            running_loss += l.item()

            # Accuracy
            top_p, top_class = torch.exp(prediction).topk(1, dim = 1)
            equals = top_class == y_batch.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Save loss and accuracy for training
        train_losses.append(running_loss / len(dl_train))
        train_acc.append((train_accuracy / len(dl_train)) * 100)    

        # Output current status on console
        print("Epoch: {:03d}/{:03d}".format(epoch + 1, epochs),
              "Training loss: {:.3f}".format(running_loss / len(dl_train)),
              "Training Accuracy: {:.3f}".format((train_accuracy / len(dl_train)) * 100))

    print("Training completed!")

    # ploting
    # if (plots == True):
    #     helpers.plot_loss_and_acc(epochs, train_losses, train_acc)

    # return model, train_losses


def evaluation(model, cuda):
    """This function performs the evaluation of the model with the test data set."""
    # Input:
    # model; the pytorch trained model

    # Load the data and put it into the DataLoader
    print("Prepare the data for validation ...")
    dataset_test = datasets_multimodal_all.WISDM_Dataset("test")
    dl_test = dataset_test.dataloading(1, False, False)
    print("Preparation of the data completed!")

    if (cuda == True):
        # moving model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(device)   # only for testing
        model = model.to(device)
    else:
        device = "cpu"

    # monitoring - evaluate test loss
    print("Start of validation ...")
    with torch.no_grad():   # no gradients, because just monitoring, no optimization

        model.eval()    # Set the model to evaluation mode
        old_id = 0
        bestpred = []
        model_name = model.__class__.__name__

        for batch in dl_test:

            new_id = int(batch[0, 0].item())
            x_batch, y_batch = batch[:, 2:5], batch[:, 1].long()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)
            top_p, top_class = torch.exp(prediction).topk(1, dim = 1)
            top_class = top_class.long()

            # Check sample_id and write value in txt if changed
            if (old_id == new_id):
                if (bestpred == []):
                    bestpred = top_class
                else:
                    bestpred = torch.cat((bestpred, top_class))
            else:
                dataset_test.writepredictions(old_id, bestpred, model_name) # Safe predictions
                bestpred = top_class

            old_id = new_id

    dataset_test.writepredictions(old_id, bestpred, model_name) # Safe last prediction

    print("Validation completed!")


def testdifhyperparameter():
    """This funciton trys different values of the hyper-parameter (user parameters) settings."""

    base_model_list = [models.MLP_NET_V1(), models.CNN_NET_V1(), models.GRU_NET(3, 4, 2, 6)]
    batch_size = [16, 64, 128, 256, 512]    # Batch size
    learning_rate = [0.01, 0.001, 0.0001]   # Learning rate

    # Test different models, batch sizes and learning rates
    for base_model in base_model_list:  # different models
        for ba in batch_size:           # different batch sizes
            for lr in learning_rate:    # different learning rates
                print("Model:", base_model.__class__.__name__, " |  Optimizer: Adam  |  Batch size:", ba, " |  Learning rate:", lr)
                model, losses = train_mlp_cnnv1_gru(base_model, 50, ba, lr, True, True)
                print(model)
                evaluation(model)
                print()


def run_train_mlp_cnnv1_gru():
    """This function performs the training and validation for the MLPm CNN V1 and GRU."""
    # The model, hyperparameters and other settings can be changed directly in this function.

    epochs = 50             # number of epochs
    batch_size = 256        # training batch size
    learning_rate = 0.001   # learning rate
    cuda = True             # true or false to train the model on cuda or not
    plots = True            # true or false to produce plots of train losses and accuracies

    mlp_v1_model = models.MLP_NET_V1()
    cnn_v1_model = models.CNN_NET_V1()
    gru_model = models.GRU_NET(3, 4, 2, 6)

    # train the model
    model, train_losses = train_mlp_cnnv1_gru(gru_model, epochs, batch_size, learning_rate, cuda, plots)
    print(model)

    # only for testing
    #print(train_losses)
    # print("Parameters of the model:", helpers.count_parameters_of_model(model))  
    torch.save(model, "model.pth")
    #model = torch.load("model.pth")
    #testdifhyperparameter()

    # evaluate the model
    evaluation(model, cuda)

#------------------------------------------------------------------------------------------------------------------#
#                                                  CNN 2D & LSTM                                                   #
#------------------------------------------------------------------------------------------------------------------#


def plot_histogram(data, name):
    _, ax = plt.subplots(figsize=(10, 3))
    ax.hist(data, bins=100, range=(-4,4))
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f"Histogram {name}")
    plt.show()


# def train_cnnv2_lstm(model, dl_train, model_type = "RNN", learning_rate = 0.1):
#     """This function trains a model (CNN 2D, LSTM) for classification using the WISDM dataset."""
#
#     # set the device which will be used to train the model
#     device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')
#     model = model.to(device)
#
#     # use CrossEntropyLoss for classification problem
#     loss = nn.CrossEntropyLoss()
#     # use SGD optimization
#     optimizer = optim.SGD(model.parameters(), lr = learning_rate)
#     # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)    #效果差
#     # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8)     #效果差
#     # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=0, eps=1e-10)     #效果和SGD相似
#     #optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)      #效果和SGD相似
#
#     train_losses = []
#     train_accuracies = []
#     best_loss = 10000.
#     best_model = model
#     lr_idx = 0
#     epoch = 0
#
#     while (learning_rate > 1e-6):
#         epoch += 1
#         # set the model in training mode
#         model.train()
#         train_loss, train_acc = 0., 0.
#
#         for batch in dl_train:
#             # send the input to the device
#             x_batch, y_batch = batch[0].to(device), batch[1].long().to(device)
#
#             if (model_type == 'CNN'):
#                 x_batch = x_batch.unsqueeze(1) # change size to [num_batch, channel, height, width]
#
#             # perform a forward pass and calculate the training loss
#             predictions = model(x_batch)
#             l = loss(predictions, y_batch)
#
#             # zero out the gradients, perform the backpropagation step, and update the weights
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_loss += l.item()*len(x_batch)
#             train_acc += (predictions.argmax(dim=1) == y_batch).type(torch.float).sum().item()
#
#         train_loss /= len(dl_train.dataset)
#         train_acc /= len(dl_train.dataset)/100.
#         train_losses.append(train_loss)
#         train_accuracies.append(train_acc)
#
#         print(f"Epoch {'{:03d}'.format(epoch)} - Training loss: {'{:.3f}'.format(train_loss)} - Training accuracy: {'{:.2f}'.format(train_acc)}% - Learning rate: {learning_rate}")
#
#         # save the best model
#         if (best_loss > train_loss):
#             best_loss = train_loss
#             best_model = model
#             lr_idx = epoch
#
#         # reduce the learning rate, if the loss has not reduced in the past epochs
#         if (lr_idx + 3 <= epoch):
#             learning_rate /= 2.
#             optimizer.param_groups[0]['lr'] = learning_rate
#             lr_idx = epoch
#             model = best_model
#
#     # plot training loss and accuracy
#     helpers.plot_loss_and_acc(epoch, train_losses, train_accuracies)
#
#     return best_model

def train_cnnv2_lstm(model, dl_train, dl_val, model_type="RNN", learning_rate=0.1):
    """This function trains a model (CNN 2D, LSTM) for classification using the WISDM dataset."""

    # set the device which will be used to train the model
    device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')
    print("设备：", device)
    model = model.to(device)

    # # 计算每个类别的样本数量
    # class_counts = [237354, 274020, 183344, 78268, 14668]
    #
    # # 计算每个类别的权重
    # total_samples = sum(class_counts)
    # class_weights = [total_samples / count for count in class_counts]
    #
    # # 将权重转换为Tensor，并根据类别数量进行归一化
    # class_weights = torch.tensor(class_weights, dtype=torch.float)
    # class_weights = class_weights / class_weights.sum()
    # class_weights = class_weights.to(device)  # 将权重张量移动到与输入数据张量相同的设备上
    #
    # # 使用权重定义损失函数
    # loss = nn.CrossEntropyLoss(weight=class_weights)

    # use CrossEntropyLoss for classification problem
    loss = nn.CrossEntropyLoss()
    # use SGD optimization
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)    #效果差
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8)     #效果差
    # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=0, eps=1e-10)     #效果和SGD相似
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)      #效果和SGD相似

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_loss = 10000.
    best_model = model
    lr_idx = 0
    epoch = 0

    while (learning_rate > 1e-6):
        epoch += 1
        # set the model in training mode
        model.train()
        train_loss, train_acc = 0., 0.

        #Train
        for batch in dl_train:
            # send the input to the device
            x_batch, y_batch = batch[0].to(device), batch[1].long().to(device)

            if (model_type == 'CNN'):
                x_batch = x_batch.unsqueeze(1)  # change size to [num_batch, channel, height, width]

            # perform a forward pass and calculate the training loss
            predictions = model(x_batch)
            l = loss(predictions, y_batch)

            # zero out the gradients, perform the backpropagation step, and update the weights
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.item() * len(x_batch)
            train_acc += (predictions.argmax(dim=1) == y_batch).type(torch.float).sum().item()

        train_loss /= len(dl_train.dataset)
        train_acc /= len(dl_train.dataset) / 100.
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_acc = 0., 0.

        with torch.no_grad():
            for batch in dl_val:
                x_batch, y_batch = batch[0].to(device), batch[1].long().to(device)

                if (model_type == 'CNN'):
                    x_batch = x_batch.unsqueeze(1)  # change size to [num_batch, channel, height, width]

                predictions = model(x_batch)
                l = loss(predictions, y_batch)
                val_loss += l.item() * len(x_batch)
                val_acc += (predictions.argmax(dim=1) == y_batch).type(torch.float).sum().item()

        val_loss /= len(dl_val.dataset)
        val_acc /= len(dl_val.dataset) / 100.
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {'{:03d}'.format(epoch)} - Training loss: {'{:.3f}'.format(train_loss)} - Training accuracy: {'{:.2f}'.format(train_acc)}% - Validation loss: {'{:.3f}'.format(val_loss)} - Validation accuracy: {'{:.2f}'.format(val_acc)}% - Learning rate: {learning_rate}")

        # save the best model
        # if (best_loss > val_loss):
        #     best_loss = val_loss
        #     best_model = model
        #     lr_idx = epoch
        if (best_loss > train_loss):
            best_loss = train_loss
            best_model = model
            lr_idx = epoch

        # reduce the learning rate, if the loss has not reduced in the past epochs
        if (lr_idx + 3 <= epoch):
            learning_rate /= 2.
            optimizer.param_groups[0]['lr'] = learning_rate
            lr_idx = epoch
            model = best_model

    # plot training loss and accuracy
    #helpers.plot_loss_and_acc(epoch, train_losses, train_accuracies, val_losses, val_accuracies)

    return best_model
# def train_cnnv2_lstm(model, dl_train, dl_val, model_type="RNN", learning_rate=0.1):
#     """This function trains a model (CNN 2D, LSTM) for classification using the WISDM dataset."""

#     # set the device which will be used to train the model
#     device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')
#     model = model.to(device)

#     # # 计算每个类别的样本数量
#     # class_counts = [237354, 274020, 183344, 78268, 14668]
#     #
#     # # 计算每个类别的权重
#     # total_samples = sum(class_counts)
#     # class_weights = [total_samples / count for count in class_counts]
#     #
#     # # 将权重转换为Tensor，并根据类别数量进行归一化
#     # class_weights = torch.tensor(class_weights, dtype=torch.float)
#     # class_weights = class_weights / class_weights.sum()
#     # class_weights = class_weights.to(device)  # 将权重张量移动到与输入数据张量相同的设备上
#     #
#     # # 使用权重定义损失函数
#     # loss = nn.CrossEntropyLoss(weight=class_weights)

#     # use CrossEntropyLoss for classification problem
#     loss = nn.CrossEntropyLoss()
#     # use SGD optimization
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#     # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)    #效果差
#     # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-8)     #效果差
#     # optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=0, weight_decay=0, eps=1e-10)     #效果和SGD相似
#     # optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-6, weight_decay=0)      #效果和SGD相似

#     train_losses = []
#     train_accuracies = []
#     val_losses = []
#     val_accuracies = []
#     best_loss = 10000.
#     best_model = model
#     lr_idx = 0
#     epoch = 0

#     while (learning_rate > 1e-6):
#         epoch += 1
#         # set the model in training mode
#         model.train()
#         train_loss, train_acc = 0., 0.

#         #Train
#         for batch in dl_train:
#             # send the input to the device
#             x1_batch = batch['EYE'].to(device)
#             x2_batch = batch['ACC'].to(device)
#             x3_batch = batch['BVP'].to(device)
#             x4_batch = batch['EDA'].to(device)
#             x5_batch = batch['TEMP'].to(device)
#             x6_batch = batch['HR'].to(device)
#             y_batch = batch['label'].long().to(device)

#             if (model_type == 'CNN'):
#                 x_batch = x_batch.unsqueeze(1)  # change size to [num_batch, channel, height, width]

#             # perform a forward pass and calculate the training loss
#             predictions = model(x1_batch, x2_batch, x3_batch, x4_batch, x5_batch, x6_batch)
#             l = loss(predictions, y_batch)

#             # zero out the gradients, perform the backpropagation step, and update the weights
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_loss += l.item() * len(x1_batch)
#             train_acc += (predictions.argmax(dim=1) == y_batch).type(torch.float).sum().item()

#         train_loss /= len(dl_train.dataset)
#         train_acc /= len(dl_train.dataset) / 100.
#         train_losses.append(train_loss)
#         train_accuracies.append(train_acc)

#         # Validation
#         model.eval()
#         val_loss, val_acc = 0., 0.

#         with torch.no_grad():
#             for batch in dl_val:
#                 x1_batch = batch['EYE'].to(device)
#                 x2_batch = batch['ACC'].to(device)
#                 x3_batch = batch['BVP'].to(device)
#                 x4_batch = batch['EDA'].to(device)
#                 x5_batch = batch['TEMP'].to(device)
#                 x6_batch = batch['HR'].to(device)
#                 y_batch = batch['label'].long().to(device)

#                 if (model_type == 'CNN'):
#                     x_batch = x_batch.unsqueeze(1)  # change size to [num_batch, channel, height, width]

#                 predictions = model(x1_batch, x2_batch, x3_batch, x4_batch, x5_batch, x6_batch)
#                 l = loss(predictions, y_batch)
#                 val_loss += l.item() * len(x1_batch)
#                 val_acc += (predictions.argmax(dim=1) == y_batch).type(torch.float).sum().item()

#         val_loss /= len(dl_val.dataset)
#         val_acc /= len(dl_val.dataset) / 100.
#         val_losses.append(val_loss)
#         val_accuracies.append(val_acc)

#         print(f"Epoch {'{:03d}'.format(epoch)} - Training loss: {'{:.3f}'.format(train_loss)} - Training accuracy: {'{:.2f}'.format(train_acc)}% - Validation loss: {'{:.3f}'.format(val_loss)} - Validation accuracy: {'{:.2f}'.format(val_acc)}% - Learning rate: {learning_rate}")

#         # save the best model
#         # if (best_loss > val_loss):
#         #     best_loss = val_loss
#         #     best_model = model
#         #     lr_idx = epoch
#         if (best_loss > train_loss):
#             best_loss = train_loss
#             best_model = model
#             lr_idx = epoch

#         # reduce the learning rate, if the loss has not reduced in the past epochs
#         if (lr_idx + 3 <= epoch):
#             learning_rate /= 2.
#             optimizer.param_groups[0]['lr'] = learning_rate
#             lr_idx = epoch
#             model = best_model

#     # plot training loss and accuracy
#     # helpers.plot_loss_and_acc(epoch, train_losses, train_accuracies, val_losses, val_accuracies)

#     return best_model


def ouput(model, time_length, batch_size, inverse_mapping_labels, model_type = "RNN"):

    # set the device which will be used to train the model
    device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')
    model = model.to(device)

    # Test data
    dataset_test = datasets_multimodal_all.WISDM_Dataset("test")
    normalized_data_test = pd.DataFrame(dataset_test.data_tensor)
    normalized_data_test.set_axis(["test-id", "subjects", "x", "y", "z"], axis = "columns", inplace = True)

    output = open("./Predictions/result.csv", "w")
    output.write('sample_id,activity\n')
    model.eval()

    for i in range(259):

        data_test = normalized_data_test.loc[normalized_data_test['test-id'] == i]
        dataset = datasets_multimodal_all.Create_Dataset(data_test[['x', 'y', 'z']], data_test[['test-id','subjects']], time_length, sliding_step=time_length)
        dl_test = DataLoader(dataset, batch_size, shuffle=False)
        y_hat = []

        for batch in dl_test:
            if (model_type == "RNN"):
                y_hat.append(model(batch[0].to(device)))
            else:
                # Shape: [samples, channel=1, height, width]
                y_hat.append(model(batch[0].unsqueeze(1).to(device)))

        y_hat = torch.cat(y_hat, dim=0)
        y_hat = y_hat.argmax(dim=1)
        output.write(f"{i},{inverse_mapping_labels[y_hat.bincount().argmax().item()]}"+"\n")

    output.close()


def run_train_cnnv2_lstm():  # 带Validation的版本
    """This function performs the training and validation for the CNN V2 and LSTM."""
    # The model, hyperparameters and other settings can be changed directly in this function.

    # List of modalities
    # modalities = ["EYE", "BVP"]
    modalities = ["EYE", "ACC", "BVP", "EDA", "TEMP", "HR"]
    # Data preprocessing
    dataset_train = datasets_multimodal_all.WISDM_Dataset("train", modalities)
    inverse_mapping_labels = dataset_train.activity_dic_inv
    mapping_labels = dataset_train.label_dic
    # Change the label from 1-5 to 0-4
    for modality in modalities:
        # dataset_train.data[modality]["Label"] = dataset_train.data[modality]["Label"].map(mapping_labels)
        dataset_train.labels[modality]["Label"] = dataset_train.labels[modality]["Label"].map(mapping_labels)

    # Plot histogram
    # plot_histogram(dataset_train.data['x'], 'x')
    # plot_histogram(dataset_train.data['y'], 'y')
    # plot_histogram(dataset_train.data['z'], 'z')
    # dataset_train.visualisation()

    # define time_length, sliding_step and batch_size for EDA/TEMP
    # time_length = 16
    # sliding_step = 8
    # batch_size = 16
    # # define time_length, sliding_step and batch_size for BVP
    # time_length = 256
    # sliding_step = 128
    # batch_size = 32
    # # define time_length, sliding_step and batch_size for HR
    # time_length = 10
    # sliding_step = 5
    # batch_size = 8
    # # define time_length, sliding_step and batch_size for ACC
    # time_length = 128
    # sliding_step = 64
    # batch_size = 32
    # define time_length, sliding_step and batch_size for EyeTrack
    # time_length = 100
    # sliding_step = 50
    # batch_size = 64
    # define time_length, sliding_step and batch_size for Multi Modalities
    time_length = 50
    sliding_step = 25
    batch_size = 64
    # dataset = datasets_multimodal_all.Create_Dataset(dataset_train.data.iloc[:, :-2], dataset_train.data[['Label']],
    #                                   dataset_train.data[['Timestamp']], time_length, sliding_step)
    dataset = datasets_multimodal_all.MultiModalityDataset(dataset_train.data, dataset_train.labels, time_length)
    # print(dataset.data_dict['EYE'].shape)
    # print(dataset.data_dict['ACC'].shape)
    for modality in modalities:
        print(dataset.data_dict[modality].shape)
    print(dataset.labels.shape)
    # sys.exit(0)

    # 定义划分比例
    train_ratio = 0.8  # 80% 划分为训练集
    val_ratio = 0.2  # 20% 划分为验证集
    # 计算划分的数量
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    # 使用 random_split 进行数据集划分
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # 分别创建 DataLoader
    dl_train = DataLoader(train_dataset, batch_size, shuffle=True)
    dl_val = DataLoader(val_dataset, batch_size, shuffle=False)

    print("OK")

    # rnn_model = models.All_Multimodal_LSTM_Attention(input_dim_1 = dataset.data_dict['EYE'].shape[2], input_dim_2 = dataset.data_dict['ACC'].shape[2],
    #                                              input_dim_3 = dataset.data_dict['BVP'].shape[2], input_dim_4 = dataset.data_dict['EDA'].shape[2],
    #                                              input_dim_5=dataset.data_dict['TEMP'].shape[2], input_dim_6=dataset.data_dict['HR'].shape[2],
    #                                              hidden_dim = 128, time_length =time_length)
    # rnn_model = models.All_Multimodal_CNN_LSTM_Hierarchical_Attention(input_dim_1=dataset.data_dict['EYE'].shape[2],
    #                                                  input_dim_2=dataset.data_dict['ACC'].shape[2],
    #                                                  input_dim_3=dataset.data_dict['BVP'].shape[2],
    #                                                  input_dim_4=dataset.data_dict['EDA'].shape[2],
    #                                                  input_dim_5=dataset.data_dict['TEMP'].shape[2],
    #                                                  input_dim_6=dataset.data_dict['HR'].shape[2],
    #                                                  hidden_dim=128, time_length=time_length)
    # print(rnn_model)
    # print("Parameters of the model:", helpers.count_parameters_of_model(rnn_model))

    # rnn_model = train_cnnv2_lstm(rnn_model, dl_train, dl_val, model_type='RNN', learning_rate = 0.1)
    # torch.save(rnn_model, "rnn_model.pth")

    # cnn_model = models.CNN_NET_V2(height=time_length, width=3)
    # print(cnn_model)
    # print("Parameters of the model:", helpers.count_parameters_of_model(cnn_model))
    #
    # cnn_model = train_cnnv2_lstm(cnn_model, dl_train, dl_val, model_type='CNN', learning_rate=0.1)
    # torch.save(cnn_model, "cnn_model.pth")

    # testing
    # ouput(rnn_model, time_length, batch_size, inverse_mapping_labels, model_type = "RNN")
    # ouput(cnn_model, time_length, batch_size, inverse_mapping_labels, model_type="CNN")


def run_train_cnnv2_lstm_allTimeSteps():  # 带Validation的版本
    """This function performs the training and validation for the CNN V2 and LSTM."""
    # The model, hyperparameters and other settings can be changed directly in this function.

    # List of modalities
    # modalities = ["EYE", "BVP"]
    modalities = ["EYE", "ACC", "BVP", "EDA", "TEMP", "HR"]
    # Data preprocessing
    dataset_train = datasets_multimodal_all.WISDM_Dataset("train", modalities)
    inverse_mapping_labels = dataset_train.activity_dic_inv
    mapping_labels = dataset_train.label_dic
    # Change the label from 1-5 to 0-4
    for modality in modalities:
        # dataset_train.data[modality]["Label"] = dataset_train.data[modality]["Label"].map(mapping_labels)
        dataset_train.labels[modality]["Label"] = dataset_train.labels[modality]["Label"].map(mapping_labels)

    # Plot histogram
    # plot_histogram(dataset_train.data['x'], 'x')
    # plot_histogram(dataset_train.data['y'], 'y')
    # plot_histogram(dataset_train.data['z'], 'z')
    # dataset_train.visualisation()

    # define time_length, sliding_step and batch_size for EDA/TEMP
    # time_length = 16
    # sliding_step = 8
    # batch_size = 16
    # # define time_length, sliding_step and batch_size for BVP
    # time_length = 256
    # sliding_step = 128
    # batch_size = 32
    # # define time_length, sliding_step and batch_size for HR
    # time_length = 10
    # sliding_step = 5
    # batch_size = 8
    # # define time_length, sliding_step and batch_size for ACC
    # time_length = 128
    # sliding_step = 64
    # batch_size = 32
    # define time_length, sliding_step and batch_size for EyeTrack
    # time_length = 100
    # sliding_step = 50
    # batch_size = 64
    # define time_length, sliding_step and batch_size for Multi Modalities
    time_length = 100
    sliding_step = 50
    batch_size = 64
    # dataset = datasets_multimodal_all.Create_Dataset(dataset_train.data.iloc[:, :-2], dataset_train.data[['Label']],
    #                                   dataset_train.data[['Timestamp']], time_length, sliding_step)
    dataset = datasets_multimodal_all.MultiModalityDataset(dataset_train.data, dataset_train.labels, dataset_train.timestamps,time_length,sliding_step)
    # print(dataset.data_dict['EYE'].shape)
    # print(dataset.data_dict['ACC'].shape)
    print(dataset.dict['EYE'].shape)
    print(dataset.dict['ACC'].shape)
    print(dataset.dict['BVP'].shape)
    print(dataset.dict['HR'].shape)
    print(dataset.dict['TEMP'].shape)
    print(dataset.dict['EDA'].shape)


    print(dataset.labels['EYE'].shape)
    print(dataset.labels['ACC'].shape)
    print(dataset.labels['BVP'].shape)
    print(dataset.labels['HR'].shape)
    print(dataset.labels['TEMP'].shape)
    print(dataset.labels['EDA'].shape)
    unique_values, counts = torch.unique(dataset.labels['ACC'], return_counts=True)
    print("Unique values in dataset.labels['ACC']: ", unique_values)
    print("Counts of each unique value: ", counts)


    # sys.exit(0)

    #合并
    combined_data = torch.cat((dataset.dict['EYE'], dataset.dict['ACC'], dataset.dict['BVP'], dataset.dict['HR'], dataset.dict['TEMP'], dataset.dict['EDA']), dim=2)
    print(combined_data.shape)
    print(dataset.labels['EYE'].shape)


    #SVM
    num_folds = 5# 交叉验证的个数
    kf = KFold(n_splits=num_folds, shuffle=True)

    for fold, (train_index, test_index) in enumerate(kf.split(combined_data)):
        print(f"Fold {fold+1}/{num_folds}")
        # 划分训练集和测试集
        train_data, test_data = combined_data[train_index], combined_data[test_index]
        train_labels, test_labels = dataset.labels['EYE'][train_index], dataset.labels['EYE'][test_index]
        
        # 将张量转换为NumPy数组
        train_data = train_data.numpy()
        test_data = test_data.numpy()
        train_labels = train_labels.numpy()
        test_labels = test_labels.numpy()

        svm_model = SVC()
        svm_model.fit(train_data.reshape(train_data.shape[0], -1), train_labels)
        svm_accuracy = svm_model.score(test_data.reshape(test_data.shape[0], -1), test_labels)
        print("SVM Accuracy:", svm_accuracy)

        # 使用随机森林进行训练和预测
        rf_model = RandomForestClassifier()
        rf_model.fit(train_data.reshape(train_data.shape[0], -1), train_labels)
        rf_accuracy = rf_model.score(test_data.reshape(test_data.shape[0], -1), test_labels)
        print("Random Forest Accuracy:", rf_accuracy)  

        # 使用XGBoost进行训练和预测
        xgb_model = XGBClassifier()
        xgb_model.fit(train_data.reshape(train_data.shape[0], -1), train_labels)
        xgb_accuracy = xgb_model.score(test_data.reshape(test_data.shape[0], -1), test_labels)
        print("XGBoost Accuracy:", xgb_accuracy) 










    # #MyLSTM模型
    # # 定义模型参数
    # # 设置超参数
    # input_size = combined_data.size(2)  # 输入特征的数量
    # hidden_size = 128
    # num_layers = 2
    # num_classes = 5

    # num_folds = 5  # 交叉验证的折数
    # # 初始化模型
    # model = models.MyLSTM(input_size, hidden_size, num_layers, num_classes)

    # # 损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # 训练模型
    # num_epochs = 10

    # # 使用交叉验证
    # kf = KFold(n_splits=num_folds, shuffle=True)

    # # 记录每个折的准确率
    # accuracy_scores = []
    # f1_scores = []
    # std_scores = []

    # for fold, (train_index, test_index) in enumerate(kf.split(combined_data)):
    #     print(f"Fold {fold+1}/{num_folds}")
    #     # 划分训练集和测试集
    #     train_data, test_data = combined_data[train_index], combined_data[test_index]
    #     train_labels, test_labels = dataset.labels['EYE'][train_index], dataset.labels['EYE'][test_index]
        
    #     for epoch in range(num_epochs):
    #         # 训练模型
    #         model.train()
    #         outputs = model(train_data)
    #         loss = criterion(outputs, train_labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            
    #         # 计算训练集的准确率
    #         _, train_predicted = torch.max(outputs, 1)
    #         train_accuracy = (train_predicted == train_labels).float().mean()
            
    #         # 测试模型
    #         model.eval()
    #         with torch.no_grad():
    #             test_outputs = model(test_data)
    #             _, test_predicted = torch.max(test_outputs, 1)
    #             accuracy = accuracy_score(test_labels, test_predicted)
    #             accuracy_scores.append(accuracy)
    #             print(f"Test Accuracy: {accuracy}")
            
    #             # 计算F1分数
    #             f1 = f1_score(test_labels, test_predicted, average='weighted')
    #             f1_scores.append(f1)
    #             print(f"Weighted F1-score: {f1}")
    #                 # 计算每个类别上的标准差
                                
    #             # 计算每个类别上的标准差
    #             class_std = []
    #             for class_label in range(num_classes):
    #                 class_diff = []
    #                 for test_label, test_pred in zip(test_labels, test_predicted):
    #                     if test_label == class_label:
    #                         class_diff.append(abs(test_label - test_pred))
    #                 if len(class_diff) > 1:  # 至少有两个样本时计算标准差
    #                     class_std.append(np.std(class_diff))
    #                 else:  # 否则将标准差设置为 0 或其他默认值
    #                     class_std.append(0.0)  # 或其他合适的默认值

    #                     # 计算每个 fold 结果的平均标准差
    #             average_std_across_classes = np.mean(class_std)
    #             std_scores.append(average_std_across_classes)

    #             print(f"Average standard deviation across classes for fold {fold+1}: {average_std_across_classes}")

    #     # 选择最佳准确率、F1分数和标准差所在的折
    # best_accuracy_fold = np.argmax(accuracy_scores)
    # best_f1_fold = np.argmax(f1_scores)
    # best_std_fold = np.mean(std_scores)

    # print(f"Best Accuracy: {accuracy_scores[best_accuracy_fold]} (Fold {best_accuracy_fold+1})")
    # print(f"Best Weighted F1-score: {f1_scores[best_f1_fold]} (Fold {best_f1_fold+1})")
    # print(f"Best Average Standard deviation:" ,best_std_fold)







    #     #CNNLSTM模型
    #     # 设置超参数
    # input_size = combined_data.size(2)  # 输入特征的数量
    # hidden_size = 128
    # num_layers = 2
    # num_classes = 5

    # num_folds = 5  # 交叉验证的折数
    #     # 初始化模型
    # model = models.CNNLSTM(input_size, hidden_size, num_layers, num_classes)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    # num_epochs = 10

    # # 使用交叉验证
    # kf = KFold(n_splits=num_folds, shuffle=True)

    # # 记录每个折的准确率
    # accuracy_scores = []
    # f1_scores = []
    # std_scores = []

    # for fold, (train_index, test_index) in enumerate(kf.split(combined_data)):
    #     print(f"Fold {fold+1}/{num_folds}")
    #     # 划分训练集和测试集
    #     train_data, test_data = combined_data[train_index], combined_data[test_index]
    #     train_labels, test_labels = dataset.labels['EYE'][train_index], dataset.labels['EYE'][test_index]
        
        
    #     for epoch in range(num_epochs):
    #         # 训练模型
    #         model.train()
    #         optimizer.zero_grad()
    #         outputs = model(train_data)
    #         loss = criterion(outputs, train_labels)
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
        
    #     # 测试模型
    #     model.eval()
    #     with torch.no_grad():
    #         test_outputs = model(test_data)
    #         _, test_predicted = torch.max(test_outputs, 1)
            
    #         # 计算准确率和F1分数
    #         accuracy = accuracy_score(test_labels, test_predicted)
    #         f1 = f1_score(test_labels, test_predicted, average='weighted')
    #         accuracy_scores.append(accuracy)
    #         f1_scores.append(f1)
    #         print(f"Test Accuracy: {accuracy}, Weighted F1-score: {f1}")
            
    #         # 计算每个类别上的标准差
    #         class_std = []
    #         for class_label in range(num_classes):
    #             class_diff = []
    #             for test_label, test_pred in zip(test_labels, test_predicted):
    #                 if test_label == class_label:
    #                     class_diff.append(abs(test_label - test_pred))
    #             if len(class_diff) > 1:  # 至少有两个样本时计算标准差
    #                 class_std.append(np.std(class_diff))
    #             else:  # 否则将标准差设置为 0 或其他默认值
    #                 class_std.append(0.0)  # 或其他合适的默认值
            
    #         # 计算每个 fold 结果的平均标准差
    #         average_std_across_classes = np.mean(class_std)
    #         std_scores.append(average_std_across_classes)

    #         print(f"Average standard deviation across classes for fold {fold+1}: {average_std_across_classes}")

    # # 选择最佳准确率、F1分数和标准差所在的折
    # best_accuracy_fold = np.argmax(accuracy_scores)
    # best_f1_fold = np.argmax(f1_scores)
    # best_std_fold = np.mean(std_scores)

    # print(f"Best Accuracy: {accuracy_scores[best_accuracy_fold]} (Fold {best_accuracy_fold+1})")
    # print(f"Best Weighted F1-score: {f1_scores[best_f1_fold]} (Fold {best_f1_fold+1})")
    # print(f"Best Average Standard deviation:" ,best_std_fold)


    



    # #GRU
    # # 定义模型参数
    # input_size = 47  # 输入特征的数量
    # hidden_size = 128
    # num_layers = 2
    # num_classes = 5

    # num_folds = 5  # 交叉验证的折数

    # # 初始化模型
    # model = models.MyGRU(input_size, hidden_size, num_layers, num_classes)

    # # 损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # 训练模型
    # num_epochs = 10

    # # 使用交叉验证
    # kf = KFold(n_splits=num_folds, shuffle=True)

    # # 记录每个折的准确率
    # accuracy_scores = []
    # f1_scores = []
    # std_scores = []

    # for fold, (train_index, test_index) in enumerate(kf.split(combined_data)):
    #     print(f"Fold {fold+1}/{num_folds}")
    #     # 划分训练集和测试集
    #     train_data, test_data = combined_data[train_index], combined_data[test_index]
    #     train_labels, test_labels = dataset.labels['EYE'][train_index], dataset.labels['EYE'][test_index]
        
    #     for epoch in range(num_epochs):
    #         # 训练模型
    #         model.train()
    #         outputs = model(train_data)
    #         loss = criterion(outputs, train_labels)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
            
    #         # 计算训练集的准确率
    #         _, train_predicted = torch.max(outputs, 1)
    #         train_accuracy = (train_predicted == train_labels).float().mean()
            
    #         # 测试模型
    #         model.eval()
    #         with torch.no_grad():
    #             test_outputs = model(test_data)
    #             _, test_predicted = torch.max(test_outputs, 1)
    #             accuracy = accuracy_score(test_labels, test_predicted)
    #             accuracy_scores.append(accuracy)
    #             print(f"Test Accuracy: {accuracy}")
                
    #             # 计算F1分数
    #             f1 = f1_score(test_labels, test_predicted, average='weighted')
    #             f1_scores.append(f1)
    #             print(f"Weighted F1-score: {f1}")
                    
    #             # 计算每个类别上的标准差
    #             class_std = []
    #             for class_label in range(num_classes):
    #                 class_diff = []
    #                 for test_label, test_pred in zip(test_labels, test_predicted):
    #                     if test_label == class_label:
    #                         class_diff.append(abs(test_label - test_pred))
    #                 if len(class_diff) > 1:  # 至少有两个样本时计算标准差
    #                     class_std.append(np.std(class_diff))
    #                 else:  # 否则将标准差设置为 0 或其他默认值
    #                     class_std.append(0.0)  # 或其他合适的默认值

    #             # 计算每个 fold 结果的平均标准差
    #             average_std_across_classes = np.mean(class_std)
    #             std_scores.append(average_std_across_classes)

    #             print(f"Average standard deviation across classes for fold {fold+1}: {average_std_across_classes}")

    # # 选择最佳准确率、F1分数和标准差所在的折
    # best_accuracy_fold = np.argmax(accuracy_scores)
    # best_f1_fold = np.argmax(f1_scores)
    # best_std_fold = np.mean(std_scores)

    # print(f"Best Accuracy: {accuracy_scores[best_accuracy_fold]} (Fold {best_accuracy_fold+1})")
    # print(f"Best Weighted F1-score: {f1_scores[best_f1_fold]} (Fold {best_f1_fold+1})")
    # print(f"Best Average Standard deviation:" ,best_std_fold)








    # #MLP
    # def train(model, criterion, optimizer, train_loader, device):
    #     model.train()
    #     total_loss = 0.0
    #     for inputs, labels in train_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     return total_loss / len(train_loader)

    # def test(model, test_loader, device):
    #     model.eval()
    #     all_preds = []
    #     all_labels = []
    #     with torch.no_grad():
    #         for inputs, labels in test_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             _, preds = torch.max(outputs, 1)
    #             all_preds.extend(preds.cpu().numpy())
    #             all_labels.extend(labels.cpu().numpy())
    #     accuracy = accuracy_score(all_labels, all_preds)
    #     f1 = f1_score(all_labels, all_preds, average='weighted')
    #     return accuracy, f1

    # # 设置超参数
    # input_size = 47  # 输入特征的数量
    # hidden_size = 128
    # num_layers = 2
    # num_classes = 5
    # num_epochs = 10
    # learning_rate = 0.001
    # batch_size = 64
    # num_folds = 5  # 交叉验证的折数

    # # 加载数据集
    # # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # 创建模型实例
    # # model = MyMLP(input_size, hidden_size, num_layers, num_classes)

    # # 损失函数和优化器
    # criterion = nn.CrossEntropyLoss()


    # # 使用交叉验证
    # kf = KFold(n_splits=num_folds, shuffle=True)

    # # 记录每个折的训练损失、测试准确率和测试F1分数
    # fold_train_losses = []
    # fold_test_accuracies = []
    # fold_test_f1s = []

    # print(combined_data.shape)
    # print(dataset.labels['EYE'].shape)
    # combined_data = torch.cat(combined_data,dataset.labels['EYE'], dim=2)
    # print(combined_data.shape)

    # for fold, (train_index, test_index) in enumerate(kf.split(combined_data)):
    #     print(f"Fold {fold+1}/{num_folds}")
        
    #     # 划分训练集和测试集
    #     train_data, test_data = combined_data[train_index], combined_data[test_index]
        
        
    #     # 创建训练和测试数据加载器
    #     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #     test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
    #     # 每个折重新初始化模型
    #     model = models.MyMLP(input_size, hidden_size, num_layers, num_classes)

    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    #     # 训练模型
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     fold_train_losses.append([])
    #     for epoch in range(num_epochs):
    #         train_loss = train(model, criterion, optimizer, train_loader, device)
    #         fold_train_losses[fold].append(train_loss)
    #         print(f"Fold {fold+1}/{num_folds}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}")
        
    #     # 在测试集上评估模型
    #     test_accuracy, test_f1 = test(model, test_loader, device)
    #     fold_test_accuracies.append(test_accuracy)
    #     fold_test_f1s.append(test_f1)
    #     print(f"Fold {fold+1}/{num_folds}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}")

    # # 计算平均测试准确率和测试F1分数
    # average_test_accuracy = sum(fold_test_accuracies) / num_folds
    # average_test_f1 = sum(fold_test_f1s) / num_folds
    # print(f"Average Test Accuracy: {average_test_accuracy}, Average Test F1: {average_test_f1}")
    #         # 这里进行训练和测试...


    # # 定义划分比例
    # train_ratio = 0.8  # 80% 划分为训练集
    # val_ratio = 0.2  # 20% 划分为验证集
    # # 计算划分的数量
    # train_size = int(train_ratio * len(dataset))
    # val_size = len(dataset) - train_size
    # # 使用 random_split 进行数据集划分
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # # 分别创建 DataLoader
    # dl_train = DataLoader(train_dataset, batch_size, shuffle=True)
    # dl_val = DataLoader(val_dataset, batch_size, shuffle=False)

    # print("OK")

    # rnn_model = models.All_Multimodal_LSTM_Attention_AllTimeSteps(input_dim_1 = dataset.data_dict['EYE'].shape[2], input_dim_2 = dataset.data_dict['ACC'].shape[2],
    #                                              input_dim_3 = dataset.data_dict['BVP'].shape[2], input_dim_4 = dataset.data_dict['EDA'].shape[2],
    #                                              input_dim_5=dataset.data_dict['TEMP'].shape[2], input_dim_6=dataset.data_dict['HR'].shape[2],
    #                                              hidden_dim = 128, time_length =time_length)
    # rnn_model = models.All_Multimodal_CNN_LSTM_Hierarchical_Attention_AllTimeSteps(input_dim_1=dataset.data_dict['EYE'].shape[2],
    #                                                               input_dim_2=dataset.data_dict['ACC'].shape[2],
    #                                                               input_dim_3=dataset.data_dict['BVP'].shape[2],
    #                                                               input_dim_4=dataset.data_dict['EDA'].shape[2],
    #                                                               input_dim_5=dataset.data_dict['TEMP'].shape[2],
    #                                                               input_dim_6=dataset.data_dict['HR'].shape[2],
    #                                                               hidden_dim=128, time_length=time_length)
    # print(rnn_model)
    # print("Parameters of the model:", helpers.count_parameters_of_model(rnn_model))

    # rnn_model = train_cnnv2_lstm(rnn_model, dl_train, dl_val, model_type='RNN', learning_rate = 0.1)
    # torch.save(rnn_model, "rnn_model.pth")


    # rnn_model = models.LSTM_NET(input_dim = dataset.data.shape[2], hidden_dim = 128, time_length =time_length)
    # print(rnn_model)
    # #print("Parameters of the model:", helpers.count_parameters_of_model(rnn_model))
    # #训练模型
    # rnn_model = train_cnnv2_lstm(rnn_model, dl_train, dl_val, model_type='RNN', learning_rate = 0.1)
    # torch.save(rnn_model, "rnn_model.pth")



    # cnn_model = models.CNN_NET_V2(height=time_length, width=3)
    # print(cnn_model)
    # print("Parameters of the model:", helpers.count_parameters_of_model(cnn_model))
    #
    # cnn_model = train_cnnv2_lstm(cnn_model, dl_train, dl_val, model_type='CNN', learning_rate=0.1)
    # torch.save(cnn_model, "cnn_model.pth")

    # testing
    # ouput(rnn_model, time_length, batch_size, inverse_mapping_labels, model_type = "RNN")
    # ouput(cnn_model, time_length, batch_size, inverse_mapping_labels, model_type="CNN")


if (__name__ == "__main__"):
    #Pr = helpers.Program_runtime()  # Calculate program runtime

    # run_train_mlp_cnnv1_gru()

    # run_train_cnnv2_lstm()

    run_train_cnnv2_lstm_allTimeSteps()

    #Pr.finish()




