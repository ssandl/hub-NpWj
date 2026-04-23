import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

dim_num = 6

# 尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。
class ClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size)
        self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, X, y = None):
        result = self.layer1(X) # (batch_size, input_size) -> (batch_size, input_size)
        y_pred = self.activation(result) # (batch_size, input_size) -> (batch_size, input_size)
        if y != None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            idx = torch.argmax(y_p)
            if idx == y_t:
                correct += 1
            else:
                wrong += 1
    print(f'正确预测个数：{correct}, 错误预测个数：{wrong}， 正确率：{correct / (correct + wrong)}')
    return correct / (correct + wrong)

def main():
    batch_size = 50
    sample_num = 5000
    lr = 0.01
    epoch = 20

    model = ClassificationModel(dim_num)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    X, Y = build_dataset(sample_num)
    for i in range(epoch):
        model.train()
        epoch_loss = []
        for j in range(sample_num // batch_size):
            x_true = X[j * batch_size : (j + 1) * batch_size]
            y_true = Y[j * batch_size : (j + 1) * batch_size]
            loss = model(x_true, y_true)
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss.append(loss.item())
        acc = evaluate(model)
        print(f'第{i + 1}轮循环中，损失为：{np.mean(epoch_loss)}，准确率为：{acc}')
        log.append([acc, np.mean(epoch_loss)])
    plt.plot(range(len(log)), [x[0] for x in log], label='acc')
    plt.plot(range(len(log)), [x[1] for x in log], label='loss')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(),"model.bin")

def build_sample():
    x = np.random.random(dim_num)
    y = np.argmax(x)
    return x, y

def build_dataset(num):
    X = []
    Y = []
    for i in range(num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

if __name__ == '__main__':
    main()
