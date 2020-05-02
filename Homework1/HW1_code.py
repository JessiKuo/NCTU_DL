# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 21:52:35 2019

@author: Kuo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def sigmoid_derv(s):
    return s * (1 - s)

def softmax(s):
    exps = np.exp(s)
    return exps / exps.sum(axis=1, keepdims=True)
#    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
#    return exps/np.sum(exps, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
    n_samples = real.shape[0]
    loss = -np.sum(np.log(pred)*real)/n_samples
    return loss

class MyNN:
    def __init__(self, x, y, lr):        
        neurons = 16
        self.lr = lr #0.01
        
        ip_dim = x.shape[1] # 6
        op_dim = y.shape[1] # 2

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        
    def feedforward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
        
    def backprop(self, x, y):
        loss = error(self.a3, y)
        
        a3_delta = cross_entropy(self.a3, y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        
        self.w1 -= self.lr * np.dot(x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)
        
        return loss

    def predict(self, train_x, train_y, test_x, test_y):
        self.feedforward(train_x)
        train_pred = np.argmax(self.a3, axis=1)
        
        cnt = 0
        for i in range(len(train_x)):
            if train_y[i] == train_pred[i]:
                cnt += 1
        train_error = 1-cnt/len(train_x)
        
        
        self.feedforward(test_x)
        test_pred = np.argmax(self.a3, axis=1)
        
        cnt = 0
        for i in range(len(test_x)):
            if test_y[i] == test_pred[i]:
                cnt += 1
        test_error = 1-cnt/len(test_x)
        
        
        return train_error, test_error

if __name__ == '__main__':
    data = pd.read_csv('./titanic.csv')
    train_y_num = (data['Survived'].iloc[:800]).values[:,np.newaxis]
    train_y = pd.get_dummies(data['Survived'].iloc[:800]).values
    train_x = (data.drop(columns=['Survived']).iloc[:800]).values
    test_y_num = (data['Survived'].iloc[800:]).values[:,np.newaxis]
    test_x = (data.drop(columns=['Survived']).iloc[800:]).values
#    del data
    
# =============================================================================
#     第3題之後，要normalize資料
# =============================================================================
#    train_x[:,5] = (train_x[:,5]-np.min(train_x[:,5]))/(np.max(train_x[:,5])- np.min(train_x[:,5]))
#    test_x[:,5] = (test_x[:,5]-np.min(test_x[:,5]))/(np.max(test_x[:,5])- np.min(test_x[:,5]))
    
    lr = 0.001
    epochs = 3000
    batch_size = 32
    
    model = MyNN(train_x, train_y, lr)
    
    L = []
    T = []
    TT = []
    
    
    for x in range(epochs):
        train_idx = np.arange(len(train_x))
        np.random.shuffle(train_idx)
        mini_batch = np.split(train_idx, batch_size)

        L_tmp = []
        
        for b in range(batch_size):
            idx = mini_batch[b]
            model.feedforward(train_x[idx])
            loss = model.backprop(train_x[idx], train_y[idx])
            L_tmp.append(loss)
        
#        print('Error :', np.mean(L_tmp))    
        
        train_error, test_error = model.predict(train_x, train_y_num, test_x, test_y_num)
        L.append(np.mean(L_tmp))
        T.append(train_error)
        TT.append(test_error)
        
    plt.plot(L)
    plt.title('training loss')
    plt.ylabel('Average cross entropy')
    plt.xlabel('epochs')
    plt.show()
    plt.close()
    
    plt.plot(T)
    plt.title('training error rate')
    plt.ylabel('Error rate')
    plt.xlabel('epochs')
    plt.show()
    plt.close()
    
    plt.plot(TT)
    plt.title('testing error rate')
    plt.ylabel('Error rate')
    plt.xlabel('epochs')
    plt.show()
    plt.close()
    
    col_name = data.columns[1:]
    
    # 4 Please identify which feature aﬀects the prediction performance the most.
    import copy
    w1 = copy.deepcopy(model.w1)
    for i in range(len(w1)):
        model.w1 = copy.deepcopy(w1)
        model.w1[i] = 0
        train_error, test_error = model.predict(train_x, train_y_num, test_x, test_y_num)
        print('把'+col_name[i]+'對應的第一層model weight設為0:\ntrain data error rate = '+\
              str(train_error)+'\n,test data error rate = ' + str(test_error))
        print('----')
    
    # 回復正常的 weight
    model.w1 = copy.deepcopy(w1)
    
    # 統計存活 sample中 pclass 跟 sex的分布
    live_idx = np.tile((train_y_num == 1), (1, 6))
    live_data = train_x[live_idx]
    live_data = live_data.reshape((int)(len(live_data)/6), 6)
    male = np.sum(live_data[:,1]==1)
    female = np.sum(live_data[:,1]==0)
    sex = ['male', 'female']
    plt.bar(sex, [male, female], alpha=0.9, width = 0.35, facecolor = 'lightskyblue', edgecolor = 'white', label='one')
    plt.show()
    
    pclass_1 = np.sum(live_data[:,0]==1)
    pclass_2 = np.sum(live_data[:,0]==2)
    pclass_3 = np.sum(live_data[:,0]==3)
    pclass = ['pclass1', 'pclass2', 'pclass3']
    plt.bar(pclass, [pclass_1, pclass_2, pclass_3], alpha=0.9, width = 0.35, facecolor = 'yellowgreen', edgecolor = 'white', label='one')
    plt.show()
    
    Val_x = [[2, 0, 25, 2, 2, 700],
             [2, 1, 25, 2, 2, 10]]
    model.feedforward(Val_x)
    train_pred = np.argmax(model.a3, axis=1)
    print(train_pred)