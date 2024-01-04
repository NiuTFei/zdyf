import tempfile
import datetime
import random
import string
from minio import Minio, S3Error
from flask import Flask, request, jsonify

import os
from torch import optim
import torch.nn as nn
import torch
import joblib
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from algorithm_module import GraphSage


def predict3(payload):
    # ============================== 一处输入
    # 获取脚本所在目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 将工作目录改为当前脚本目录
    os.chdir(script_dir)

    ## 载入数据
    graph_data_dict = joblib.load('./data/graph_data_dict.pkl')
    # ==============================

    ## 超参数
    ratio = 0.3  # 测试集所占比例
    batch = 100  # 批量大小
    input_feature = 2048  # 输入维度
    out_feature = 5  # 输出维度
    init_lr = 0.001  # 初始学习率
    total_epoch = 50  # 训练多少轮次

    ## 划分训练集、测试集
    train_data = []
    test_data = []

    for key in graph_data_dict.keys():
        tmp_graph_list = graph_data_dict[key]
        tmp_train_normal, tmp_test_normal = train_test_split(tmp_graph_list, test_size=ratio, random_state=0)

        train_data += tmp_train_normal
        test_data += tmp_test_normal

    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    ## 确定device
    if torch.cuda.is_available():
        print('device = gpu')
        device = torch.device("cuda")
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")

    model = GraphSage(feature=input_feature, out_channel=out_feature).to(device)

    ## 定义优化器、损失函数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr)  # 优化器
    criterion = nn.CrossEntropyLoss()  # 损失函数

    ## 训练
    # 记录
    batch_count = 0  #
    batch_loss = 0.0
    batch_acc = 0.0

    train_acc = []  # 记录训练过程中，训练集上acc变化
    test_acc = []  # 记录训练过程中，测试集上acc变化

    train_loss = []  # 记录训练集上损失
    test_loss = []  # 记录测试集上损失

    for epoch in range(total_epoch):
        # 记录当前训练轮次的acc和loss
        epoch_acc = 0
        epoch_loss = 0.0
        sample_num = 0

        # 训练mode
        model.train()
        for data in train_loader:  # 按照batch size大小分批训练，直到训练集过完一遍算是完成一个epoch，而这里每次循环是一个step
            inputs = data.to(device)
            labels = inputs.y

            batch_num = inputs.num_graphs
            sample_num += len(labels)

            with torch.set_grad_enabled(True):  # 允许梯度变化
                # 将inputs送入模型，得到结果，即前向传播
                logits = model(inputs)

                # 计算该step的平均损失
                loss = criterion(logits, labels.long())
                # 计算准确率
                pred = logits.argmax(dim=1)  # 预测标签
                correct = torch.eq(pred, labels).float().sum().item()  # 正确分类的样本总数
                epoch_acc += correct  # 一个epoch总的分类正确的样本数量等于各个step分类正确的数目之和
                # 计算该step的总的损失，用epoch_loss等于各个step损失之和
                loss_temp = loss.item() * batch_num
                epoch_loss += loss_temp

                # 反向传播，更新梯度
                optimizer.zero_grad()  # 清空之前的梯度值
                loss.backward()
                optimizer.step()

                # 打印信息
                batch_loss += loss_temp  # 该batch的loss
                batch_acc += correct  # 该batch的acc
                batch_count += batch_num  # 该batch的样本总量

                batch_loss = batch_loss / batch_count
                batch_acc = batch_acc / batch_count

                # print(('Epoch: {}, Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, batch_loss, batch_acc)))
                batch_acc = 0
                batch_loss = 0.0
                batch_count = 0

        # 整个batch过一遍，相当于完成了一个epoch，打印epoch信息
        epoch_loss = epoch_loss / sample_num
        epoch_acc = epoch_acc / sample_num
        print(('Epoch: {}/{}-Loss: {:.4f} {}-Acc: {:.4f}'.format(epoch, total_epoch, epoch_loss, 'Train', epoch_acc)))
        # 记录，画图备用
        train_acc.append(epoch_acc)
        train_loss.append(epoch_loss)

        # 更新学习率
        lr_adjust = {30: 5e-4, 40: 1e-4}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('=' * 20, 'Updating learning rate to {}'.format(lr))

    ## 测试
    for data in test_loader:
        inputs = data.to(device)
        labels = inputs.y
        logits = model(inputs)
        pred = logits.argmax(dim=1)
        correct = torch.eq(pred, labels).float().sum().item()  # 正确分类的样本总数
        fc_features = model.fc_feature
        test_acc = correct / inputs.num_graphs

    print('测试集上准确率: {:.4f}'.format(test_acc))

    ## 绘图
    # 准确率、损失函数变化曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure()
    plt.plot(train_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Acc/%')
    plt.title('训练集准确率变化曲线')

    plt.figure()
    plt.plot(train_loss, color='y')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('损失函数变化曲线')
    plt.show()

    # t-sne
    feature = fc_features.cpu().detach().numpy()
    true_label = labels.cpu().numpy().astype('int')
    pred_label = pred.cpu().numpy().astype('int')
    tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=50, n_iter=2000, learning_rate=500)
    X = tsne.fit_transform(feature)

    # 混淆矩阵
    sns.set()
    sns.set_style("white")
    color_list = ['#fc7d77', '#aef76a', '#7290fc', '#b482fa', '#fca265']
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=true_label, marker="o", palette=color_list)
    C = confusion_matrix(true_label, pred_label, labels=[0, 1, 2, 3, 4], normalize='true')
    plt.matshow(C, cmap=plt.cm.tab10)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate('%.2f' % C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # 由于可视化部分封装，此处只返回测试集准确率。可根据需要进行调整，将可视化部分独立，result返回绘图所需要的变量以供后续调用
    result = {'Test_acc': test_acc}

    return result

