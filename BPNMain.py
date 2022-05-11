# coding = utf-8
'''

date :  2022-4-27
desc:   BP神经网络学习案例

'''

import numpy as np
import math
import random
#import string
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import re

# random.seed(0)  #当我们设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数
# 参考https://blog.csdn.net/jiangjiang_jian/article/details/79031788

# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a


# 生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)
    return a


# 函数sigmoid(),这里采用tanh，因为看起来要比标准的sigmoid函数好看
def sigmoid(x):
    return math.tanh(x)


# 函数sigmoid的派生函数
def derived_sigmoid(x):
    return 1.0 - x ** 2


# 构造三层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层，输出层的节点数
        self.num_in = num_in + 1  # 增加一个偏置结点
        self.num_hidden = num_hidden + 1  # 增加一个偏置结点
        self.num_out = num_out

        # 激活神经网络的所有节点（向量）
        self.active_in = [1.0] * self.num_in
        self.active_hidden = [1.0] * self.num_hidden
        self.active_out = [1.0] * self.num_out

        # 创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden)
        self.wight_out = makematrix(self.num_hidden, self.num_out)

        # 对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.wight_in[i][j] = random_number(-0.2, 0.2)
                #self.wight_in[i][j] = random_number(-0.2, 0.2)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.wight_out[i][j] = random_number(-0.2, 0.2)

        # 最后建立动量因子（矩阵）
        self.ci = makematrix(self.num_in, self.num_hidden)
        self.co = makematrix(self.num_hidden, self.num_out)

        # 信号正向传播

    def update(self, inputs):
        if len(inputs) != self.num_in - 1:
            raise ValueError('与输入层节点数不符')

        # 数据输入输入层
        for i in range(self.num_in - 1):
            # self.active_in[i] = sigmoid(inputs[i])  #或者先在输入层进行数据处理
            self.active_in[i] = inputs[i]  # active_in[]是输入数据的矩阵

        # 数据在隐藏层的处理
        for i in range(self.num_hidden - 1):
            sum = 0.0
            for j in range(self.num_in):
                sum = sum + self.active_in[i] * self.wight_in[j][i]
            self.active_hidden[i] = sigmoid(sum)  # active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据

        # 数据在输出层的处理
        for i in range(self.num_out):
            sum = 0.0
            for j in range(self.num_hidden):
                sum = sum + self.active_hidden[j] * self.wight_out[j][i]
            self.active_out[i] = sigmoid(sum)  # 与上同理

        return self.active_out[:]

    # 误差反向传播
    def errorbackpropagate(self, targets, lr, m):  # lr是学习率， m是动量因子
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')

        # 首先计算输出层的误差
        out_deltas = [0.0] * self.num_out
        for i in range(self.num_out):
            error = targets[i] - self.active_out[i]
            out_deltas[i] = derived_sigmoid(self.active_out[i]) * error

        # 然后计算隐藏层误差
        hidden_deltas = [0.0] * self.num_hidden
        for i in range(self.num_hidden):
            error = 0.0
            for j in range(self.num_out):
                error = error + out_deltas[j] * self.wight_out[i][j]
            hidden_deltas[i] = derived_sigmoid(self.active_hidden[i]) * error

        # 首先更新输出层权值
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                change = out_deltas[j] * self.active_hidden[i]
                self.wight_out[i][j] = self.wight_out[i][j] + lr * change + m * self.co[i][j]
                self.co[i][j] = change

        # 然后更新输入层权值
        for i in range(self.num_in):
            for i in range(self.num_hidden):
                change = hidden_deltas[j] * self.active_in[i]
                self.wight_in[i][j] = self.wight_in[i][j] + lr * change + m * self.ci[i][j]
                self.ci[i][j] = change

        # 计算总误差
        error = 0.0
        for i in range(len(targets)):
            error = error + 0.5 * (targets[i] - self.active_out[i]) ** 2
        return error

    # 测试
    def test(self, patterns):
        for i in patterns:
            #print(i[0], '->', self.update(i[0]))
            print(i, '->', self.update(i[0]))
    # 权重
    def weights(self):
        print("输入层权重")
        for i in range(self.num_in):
            print(self.wight_in[i])
        print("输出层权重")
        for i in range(self.num_hidden):
            print(self.wight_out[i])

    def train(self, pattern, itera=5000, lr=0.001, m=0.1):
        for i in range(itera):
            error = 0.0
            for j in pattern:
                inputs = j[0]
                targets = j[1]
                self.update(inputs)
                error = error + self.errorbackpropagate(targets, lr, m)
            if i % 100 == 0:
                print('误差 %-.5f' % error)


# 实例
def demo():
    '''
    patt = [
        [[1, 2, 5], [0]],
        [[1, 3, 4], [1]],
        [[1, 6, 2], [1]],
        [[1, 5, 1], [0]],
        [[1, 8, 4], [1]],
        [[-1,-5,4],[-1]]
    ]

    patt = [
        [[8.0, 0.0, 8.0], [5.0]],
        [[1.8, 1.0, 2.8], [6.0]],
        [[7.0, 2.0, 9.0], [-5.0]],
        [[4.333333333333333, 3.0, 7.333333333333333], [17.0]],
        [[4.666666666666667, 4.0, 8.666666666666668], [-10.0]],
        [[5.0, 5.0, 10.0], [25.0]],
        [[5.333333333333333, 6.0, 11.333333333333332], [-22.0]],
        [[5.666666666666667, 7.0, 12.666666666666668], [33.0]],
        [[2.25, 8.0, 10.25], [-28.0]],
        [[2.375, 9.0, 11.375], [37.0]],
        [[2.5, 6.0, 8.5], [-34.0]]
    ]
    
    '''
    p1=[]
    p2=[]
    patt = []

    f = open('G:/python_test/HeadPoseEstimation/Xtestdata.txt','r')
    fX = open('G:/python_test/HeadPoseEstimation/Xtestdata-1.txt','w')

    #f = open('G:/python_test/HeadPoseEstimation/Ytestdata.txt','r')
    #fY = open('G:/python_test/HeadPoseEstimation/Ytestdata-1.txt','w')
    line = f.readline()

    while line:
        p = re.compile(r'\d+')
        r1 = p.findall(line)

        for i in range(9):
            p1.append(int(r1[i]))
        p2.append(int(r1[9]))
        p3 = [p1,p2]
        patt.append(p3)
        p1=[]
        p2=[]
        line = f.readline()
    #数据归一处理
    xaverage = []
    s = []
    x = 0
    s1 = 0
    for j in range(9):
        '''
        #标准化处理

        i = 0
        for i in range(len(patt)):
            x += patt[i][0][j]
        xaverage.append( x / len(patt))
        i = 0
        for i in range(len(patt)):
            s1 += (patt[i][0][j]-xaverage[j])**2
        s.append((s1/len(patt))**0.5)
        i = 0
        for i in range(len(patt)):
            patt[i][0][j] = (patt[i][0][j]-xaverage[j])/s[j]
        '''
        #Min-Max处理
        max = 0
        min = 0
        for i in range(len(patt)):
            if max < patt[i][0][j]:
                max = patt[i][0][j]
            if min > patt[i][0][j]:
                min = patt[i][0][j]
        for i in range(len(patt)):
            patt[i][0][j] = (patt[i][0][j]-min)/(max-min)
    for i in range(len(patt)):
        for j in range(9):
            fX.write(str(patt[i][0][j])+' ')
        fX.write(str(patt[i][1][0]))
        fX.write('\n')
    #print(patt[0])

    f.close()
    '''
    # 创建神经网络，3个输入节点，3个隐藏层节点，1个输出层节点
    n = BPNN(9, 3, 1)
    # 训练神经网络
    n.train(patt)
    # 测试神经网络
    n.test(patt)
    # 查阅权重值
    n.weights()
    '''

if __name__ == '__main__':
    demo()
