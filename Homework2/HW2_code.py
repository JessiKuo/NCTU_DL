# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:37:20 2019

@author: Kuo
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class utilsTools:
    
    @staticmethod
    def dataPreprocessing():
        '''
        HW2-1 : Convolutional Neural Network for Image Recognition
        資料前處理，只有resize圖片，不另外考慮圖片變形問題
        '''
        folder = './animal-10/'
        target_folder = './animal-10-px50/'
        
        sub_folder = ['train', 'val']
        
        # 全部要處理的資料有分成 train, val
        for i in range(len(sub_folder)):
            path = folder + sub_folder[i] + '/'
            sub_sub_folder = os.listdir(path)
            
            # 各有 10 種類別
            for j in range(len(sub_sub_folder)):
                path_tmp = path + sub_sub_folder[j] + '/'
                pics = os.listdir(path_tmp)
                
                # 處理好的資料要儲存到 store_folder 指定的位置
                store_folder = target_folder + sub_folder[i] + '/' + \
                                sub_sub_folder[j] + '/'
                if not os.path.exists(store_folder):
                    os.makedirs(store_folder)
                
                # 每個類別有 1000 張圖片
                for p in range(len(pics)): 
                    file_tmp = path_tmp + pics[p]
                    img = Image.open(file_tmp)
                    
                    img = img.resize((50, 50), Image.ANTIALIAS)
                    img = img.convert('RGB')
                    img.save(store_folder + str(p) + '.jpg')
    @staticmethod
    def readData(no):
        if no == 1:
            '''
            HW2_1 圖片圖檔
            '''
            data_folder = './animal-10-clean/'
            category = os.listdir(data_folder + 'train/')
            
            # training data共有10000筆 data，因為有10類每類1000張圖片
            trainData = np.empty((10000, 2), dtype = object)
            for i in range(len(category)):
                path_tmp = data_folder + 'train/' + category[i] + '/'
                label = np.zeros(len(category))
                label[i] = 1
                
                for j in range(1000):
                    img = Image.open(path_tmp + str(j) + '.jpg')
                    img = np.array(img)
                    img = np.swapaxes(img, 0, 2)
        #            mmm = Image.fromarray(np.swapaxes(img, 2, 0))
                    
                    trainData[i*1000+j][0] = img
                    trainData[i*1000+j][1] = i
                    
            valData = np.empty((4000, 2), dtype=object)
            for i in range(len(category)):
                path_tmp = data_folder + 'val/' + category[i] + '/'
                label = np.zeros(len(category))
                label[i] = 1
                
                for j in range(400):
                    img = Image.open(path_tmp + str(j) + '.jpg')
                    img = np.array(img)
                    img = np.swapaxes(img, 0, 2)
                    
                    valData[i*400+j][0] = img
                    valData[i*400+j][1] = i
            
    #        np.random.shuffle(trainData)
    #        np.random.shuffle(valData)
            
            dictOfCategory = { i : category[i] for i in range(0, len(category) ) }
            
            return np.array(trainData[:,0]), np.array(trainData[:,1]), \
                    np.array(valData[:,0]), np.array(valData[:,1]), dictOfCategory
        
        elif no == 2:
            '''
            HW2_2 文件讀檔
            '''
            accept = pd.read_excel('./ICLR_accepted.xlsx')
            reject = pd.read_excel('./ICLR_rejected.xlsx')
            
            train = accept[50:]
            train = train.append(reject[50:])
            train = train.values
            
            word = set()
            for i in range(len(train)):
                tmp_s = (train[i][0]).lower()
#                tmp_s = tmp_s.translate({ord(ch):' ' for ch in ':,&$#^/?\\'})
                tmp_s = tmp_s.split(' ')
                tmp_set = set(tmp_s)
                word = word.union(tmp_set)
            
            word = list(word)
            dictionary = {k: v for v, k in enumerate(word)}
            
#            dictionary['<unk>'] = 0
            
            accept_clean = list()
            for i in range(len(accept)):
                tmp_s = accept.iloc[i][0].lower().split(' ')
                ss = list()
                for w in range(len(tmp_s)):
                    if tmp_s[w] in dictionary:
                        ss.append(dictionary[tmp_s[w]])
#                        ss += (str(dictionary[tmp_s[w]]) + ' ')
                    else:
                        ss.append(dictionary[''])
#                        ss += '1 '
                accept_clean.append([ss])
                
                
            reject_clean = list()
            for i in range(len(reject)):
                tmp_s = reject.iloc[i][0].lower().split(' ')
                ss = list()
                for w in range(len(tmp_s)):
                    if tmp_s[w] in dictionary:
                        ss.append(dictionary[tmp_s[w]])
                    else:
                        ss.append(dictionary[''])
                reject_clean.append([ss])
            
            accept_clean = pd.DataFrame(accept_clean)
            reject_clean = pd.DataFrame(reject_clean)
            
            accept_clean['label'] = [[0]]*len(accept) # [1, 0] : accept
            reject_clean['label'] = [[1]]*len(reject) # [0, 1] : reject
        
            testData = accept_clean[:50]
            testData = testData.append(reject_clean[:50])
            
            trainData = accept_clean[50:]
            trainData = trainData.append(reject_clean[50:])
            
            trainX = trainData.iloc[:,0].tolist()
            testX = testData.iloc[:,0].tolist()
            
            zero = np.zeros(10, dtype=int)
            for i in range(len(trainX)):
                length = len(trainX[i])
                if  length >= 10:
                    trainX[i] = trainX[i][:10]
                else:
                    trainX[i].extend(zero[:10-length])
            
            for i in range(len(testX)):
                length = len(testX[i])
                if  length >= 10:
                    testX[i] = testX[i][:10]
                else:
                    testX[i].extend(zero[:10-length])
            
            
            return trainX, trainData.iloc[:,1].tolist(), \
                   testX , testData.iloc[:,1].tolist(), dictionary
            
        else:
            print('Read fail!')
            return 
    
    @staticmethod
    def drawFigure(data, x_label, ylabel, title, legend, legendBool):
        for i in range(len(data)):
            plt.plot(data[i], label=legend[i])
            
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.title(title)
        if legendBool: plt.legend()
        plt.show()
        
    @staticmethod
    def HW2_1_test_performance(valX, valY, cnn, dictOfCategory):
        # 隨機選擇 9 張照片
        idx = np.random.randint(len(valX), size = [9])
        x = valX[idx]
        y = valY[idx]
        
        val_x = torch.from_numpy(np.array(list(x)))
        output = cnn(val_x.double())[0] 
        pred_y = torch.max(output, 1)[1].data.numpy()
        
        # 因為圖片有被正規化過，要轉回來
        x*=255
    
        plt.figure(figsize=(6, 6))
        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.tight_layout()
            
            mmm = Image.fromarray(np.swapaxes(x[i], 2, 0).astype(np.uint8))
            plt.imshow(mmm, interpolation='none')
            plt.title("pred: {},\ntruth: {}".format(dictOfCategory[pred_y[i]],
                                                dictOfCategory[y[i]]), fontsize = 10)
            plt.xticks([])
            plt.yticks([])
        
        plt.show()
        
        cat_acc = []
        for i in range(len(dictOfCategory)):
            tmp_idx = np.arange(i*400, i*400+400)
            
            val_x = torch.from_numpy(np.array(list(valX[tmp_idx])))
            output = cnn(val_x.double())[0] 
            pred_y = torch.max(output, 1)[1].data.numpy()
            cnt = float((pred_y == valY[tmp_idx]).astype(int).sum())
            
            cat_acc.append(cnt/4)
            
        print('Accuracy of classes')
        for i in range(len(dictOfCategory)):
            print("%-10s : %s %%" % (dictOfCategory[i], str(cat_acc[i])))
        
        

class CNN(nn.Module):
    def __init__(self, FILTER, STRIDE, KERNEL):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=FILTER[0], #16            
                kernel_size=KERNEL[0], # 5              
                stride=STRIDE[0], # 1                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),   
        )
        self.conv2 = nn.Sequential(  
            nn.Conv2d(FILTER[0], FILTER[1], KERNEL[1], STRIDE[1], 2),    
            nn.ReLU(),                    
            nn.MaxPool2d(2),               
        )
        tmp = int((100+2*2-KERNEL[0])/STRIDE[0])+1
        tmp = tmp / 2 # pooling
        if tmp-int(tmp) > 0.5:
            tmp = int(tmp)+1
        else:
            tmp = int(tmp)
        
        tmp = int((tmp+2*2-KERNEL[1])/STRIDE[1])+1
        tmp = tmp / 2 # pooling
        if tmp-int(tmp) > 0.5:
            tmp = int(tmp)+1
        else:
            tmp = int(tmp)
        
        self.out = nn.Linear(FILTER[1] * tmp * tmp, 10)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           
        output = self.out(x)
        output = F.softmax(output, dim=1)
        return output, x   

class HW:
    
    def __init__(self):
        pass
    
    @staticmethod
    def HW2_1(trainX,
              trainY,
              valX,
              valY,
              EPOCH,
              GROUP_SIZE,
              LR,
              FILTER,
              STRIDE,
              KERNEL
              ):
        
        # data normalization
        trainX /= 255
        valX /= 255
        
        cnn = CNN(FILTER, STRIDE, KERNEL)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
        loss_func = nn.CrossEntropyLoss()
        
        trainLoss = []
        trainAcc = []
        valAcc = []
        
        cnn = cnn.double()
        for epoch in range(EPOCH):
            train_idx = np.arange(len(trainX))
            np.random.shuffle(train_idx)
            group = np.split(train_idx, GROUP_SIZE) 
            
            for g in range(len(group)):
                
                images_batch = torch.from_numpy(np.array(list(trainX[group[g]])))
                labels_batch = torch.tensor(np.array(list(trainY[group[g]])), dtype=torch.long)
                
                output = cnn(images_batch.double())[0]        
                loss = loss_func(output, labels_batch.long())
                optimizer.zero_grad()           
                loss.backward()                 
                optimizer.step()                
                
            trainLoss.append(loss.data.numpy())
            
            cnt = 0
            for g in range(len(group)):
                train_x = torch.from_numpy(np.array(list(trainX[group[g]])))
                output = cnn(train_x.double())[0] 
                pred_y = torch.max(output, 1)[1].data.numpy()
                cnt += float((pred_y == trainY[group[g]]).astype(int).sum())
            
            train_accuracy = cnt / float(len(trainY))
            
            cnt = 0
            val_idx = np.arange(len(valX))
            group = np.split(val_idx, GROUP_SIZE)
            for g in range(len(group)):
                val_x = torch.from_numpy(np.array(list(valX[group[g]])))
                output = cnn(val_x.double())[0] 
                pred_y = torch.max(output, 1)[1].data.numpy()
                cnt += float((pred_y == valY[group[g]]).astype(int).sum())
            
            val_accuracy = cnt / float(len(valY))
        
            trainAcc.append(train_accuracy)
            valAcc.append(val_accuracy)
        
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(),\
                   '| train accuracy: %.2f' % train_accuracy, 
                   '| test accuracy: %.2f' % val_accuracy
                 )
            
        return trainLoss, trainAcc, valAcc, cnn
        
    @staticmethod
    def HW2_2():
        pass

#from torchsummary import summary
import torch.utils.data as data_utils

#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, rnn_model, 
                 hidden_size, num_layers):
       
        super(RNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        elif rnn_model == 'RNN':
            self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size)

        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_output)

    def forward(self, x):
        x_embed = self.embed(x.t())

        packed_output, ht = self.rnn(x_embed)

        out = F.relu(self.fc1(packed_output[-1]))
        out = self.fc2(out)
        
        return out
    
    @staticmethod
    def train(model, num_epochs, BATCH_SIZE, train_data_dataloader, test_data_dataloader, \
              trainX, trainY, testX, testY):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        Loss = []
    
        trainAcc = []
        trainError = []
        testAcc = []
        testError = []
        
        for epoch in range(num_epochs):
            ave_loss = 0
            for i, (x, y) in enumerate(train_data_dataloader):
                x = x.to(torch.long)
                y = y.to(torch.long).squeeze_()
                
                output = model(x)
                loss = criterion(output, y)
                ave_loss += loss.item()*x.size(0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            ll = ave_loss/len(trainX)
            Loss.append(ll)
            
            cnt = 0
            for batch_idx, (x, y) in enumerate(test_data_dataloader):
                x, y = x.to(torch.long), y.to(torch.long)
                out = model(x)
                pred_y = torch.max(out, 1)[1].data.numpy()
                pred_y = pred_y.reshape(-1, 1)
                cnt += float((pred_y == np.array(y)).astype(int).sum())
            test_accuracy = cnt / float(len(testY))
            
            
            cnt = 0
            for batch_idx, (x, y) in enumerate(train_data_dataloader):
                x, y = x.to(torch.long), y.to(torch.long)
                out = model(x)
                pred_y = torch.max(out, 1)[1].data.numpy()
                pred_y = pred_y.reshape(-1, 1)
                cnt += float((pred_y == np.array(y)).astype(int).sum())
            train_accuracy = cnt / float(len(trainY))
            
            
            trainAcc.append(train_accuracy)
            trainError.append(1-train_accuracy)
            testAcc.append(test_accuracy)
            testError.append(1-test_accuracy)
                
            print('Epoch: ', epoch, '| train loss: %.4f' % ll,\
                   '| train accuracy: %.2f' % train_accuracy, 
                   '| test accuracy: %.2f' % test_accuracy
                 )
            
        return Loss,trainAcc, testAcc, trainError, testError

if __name__ == '__main__':
    
# =============================================================================
#     第一題
# =============================================================================
    trainX, trainY, valX, valY, dictOfCategory = utilsTools.readData(1)
    
    EPOCH = 3
    GROUP_SIZE = 20 
    LR = 0.001
    
#     param setting 1
    FILTER = [16, 32]
    STRIDE = [2, 2]
    KERNEL = [5, 5]
    trainLoss, trainAcc, valAcc, cnn = HW.HW2_1(trainX, trainY, valX, valY, \
                                           EPOCH, GROUP_SIZE, LR, FILTER, STRIDE, KERNEL)
    utilsTools.drawFigure([trainLoss], 'Number of epochs', 'Cross entropy', \
                          'Learning curve', ['loss'], False)
    utilsTools.drawFigure([trainAcc, valAcc], 'Number of epochs', 'Accuracy rate', \
                          'Training Accuracy', ['train', 'test'], True)
    
    utilsTools.HW2_1_test_performance(valX, valY, cnn, dictOfCategory)
    
#     param setting 2
    FILTER = [16, 32]
    STRIDE = [2, 2]
    KERNEL = [10, 5]
    trainLoss, trainAcc, valAcc = HW.HW2_1(EPOCH, GROUP_SIZE, LR, FILTER, STRIDE, KERNEL)
    utilsTools.drawFigure([trainLoss], 'Number of epochs', 'Cross entropy', \
                          'Learning curve', ['loss'], False)
    utilsTools.drawFigure([trainAcc, valAcc], 'Number of epochs', 'Accuracy rate', \
                          'Training Accuracy', ['train', 'test'], True)
    utilsTools.HW2_1_test_performance(valX, valY, cnn, dictOfCategory)
    
    # =============================================================================
#     第二題
# =============================================================================
    num_epochs = 1400
    BATCH_SIZE = 32
    
    trainX, trainY, testX, testY, dictionary = utilsTools.readData(2)
    
    tensor_x_train = torch.stack([torch.Tensor(i) for i in trainX])
    tensor_y_train = torch.stack([torch.Tensor(i) for i in trainY])
    train_data = data_utils.TensorDataset(tensor_x_train, tensor_y_train)
    train_data_dataloader = data_utils.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True) 

    tensor_x_test = torch.stack([torch.Tensor(i) for i in testX])
    tensor_y_test = torch.stack([torch.Tensor(i) for i in testY])
    test_data = data_utils.TensorDataset(tensor_x_test, tensor_y_test)
    test_data_dataloader = data_utils.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle=True) 
    
    # RNN
    model = RNN(vocab_size=len(dictionary), embed_size = 100, num_output = 2, \
        rnn_model = 'RNN', hidden_size = 64, num_layers = 3)#.to(device)
    
    Loss,trainAcc, testAcc, trainError, testError = \
        RNN.train(model, num_epochs, BATCH_SIZE, train_data_dataloader, \
                  test_data_dataloader, trainX, trainY, testX, testY)
    
    utilsTools.drawFigure([Loss], 'Number of epochs', 'Cross entropy', \
                           'Learning curve', ['loss'], False)
    
    utilsTools.drawFigure([trainAcc, testAcc], 'Number of epochs', 'Accuracy rate', \
                           'RNN Accuracy Curve', ['train', 'test'], True)
    
    utilsTools.drawFigure([trainError, testError], 'Number of epochs', 'Error rate', \
                           'RNN Error Curve', ['train', 'test'], True)
    
    # LSTM
    model = RNN(vocab_size=len(dictionary), embed_size = 100, num_output = 2, \
        rnn_model = 'LSTM', hidden_size = 64, num_layers = 3)#.to(device)
    
    
    Loss,trainAcc, testAcc, trainError, testError = \
        RNN.train(model, num_epochs, BATCH_SIZE, train_data_dataloader, \
                  test_data_dataloader, trainX, trainY, testX, testY)
    
    utilsTools.drawFigure([Loss], 'Number of epochs', 'Cross entropy', \
                           'Learning curve', ['loss'], False)
    
    utilsTools.drawFigure([trainAcc, testAcc], 'Number of epochs', 'Accuracy rate', \
                           'LSTM Accuracy Curve', ['train', 'test'], True)
    
    utilsTools.drawFigure([trainError, testError], 'Number of epochs', 'Error rate', \
                           'LSTM Error Curve', ['train', 'test'], True)
    
    
    
    
    
        
    
    
    
    
    
    
    
