import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import time
import scipy.io as scio
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import CNN
import matplotlib.pyplot as plt

print("Reading data")
dataFile2 = './data/MNIST.mat'
data2 = scio.loadmat(dataFile2)
# print(data2.keys())

train = data2['fea']
print("Shape of data = {}".format(train.shape))
train=train.reshape(70000,1,28,28)
print("After transfering,shape of data = {}".format(train.shape))
# print(train[0])
label = data2['gnd']
# print(label.shape)
train_x = train[:50000]
train_y = label[:50000]
print("Size of training data = {}".format(len(train_x)))
# print("Size of training data = {}".format(len(train_y)))
val_x = train[50000:65000]
val_y = label[50000:65000]
print("Size of validation data = {}".format(len(val_x)))
# print(ts_x.shape)
test_x = train[65000:68000]
test_y = label[65000:68000]
test_y=test_y.reshape(3000)
# print(test_y)
print("Size of Testing data = {}".format(len(test_x)))
print("reading over")

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

batch_size = 500
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_set = ImgDataset(train_x, train_y)
val_set = ImgDataset(val_x, val_y)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,num_workers=num_workers)

# start = time.time()
# for X, y in train_loader:
#     continue
# print('%.2f sec' % (time.time() - start))

net = CNN.Classifier()#实例化网络
for parameters in net.parameters():#打印出参数矩阵及值
    print(parameters)
for name, parameters in net.named_parameters():#打印出每一层的参数的大小
    print(name, ':', parameters.size())
# for param_tensor in net.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
#     print(param_tensor, '\t', net.state_dict()[param_tensor].size())
train_loss_list=[]
val_loss_list = []
train_acc_list = []
val_acc_list = []
#training
model = CNN.Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 因為是classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 25
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        data[0] = data[0].type(torch.FloatTensor)
        data[1]=data[1].reshape(500)
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data[0] = data[0].type(torch.FloatTensor)
            data[1] = data[1].reshape(500)
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()


        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, \
               train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(),
               val_loss / val_set.__len__()))
        train_loss_list.append(train_loss / train_set.__len__())
        val_loss_list.append(val_loss / val_set.__len__())
        train_acc_list.append(train_acc / train_set.__len__())
        val_acc_list.append(val_acc / val_set.__len__())

plt.plot(np.arange(num_epoch),train_loss_list)
plt.plot(np.arange(num_epoch),val_loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(np.arange(num_epoch),train_acc_list)
plt.plot(np.arange(num_epoch),val_acc_list)
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.show()

train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)
train_val_set = ImgDataset(train_val_x, train_val_y)
train_val_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True)
train_loss_list1=[]
train_acc_list1=[]

model_best = CNN.Classifier().cuda()
loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001) # optimizer 使用 Adam
num_epoch = 25
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model_best.train()
    for i, data in enumerate(train_val_loader):
        data[0] = data[0].type(torch.FloatTensor)
        data[1] = data[1].reshape(500)
        optimizer.zero_grad()
        train_pred = model_best(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

        #將結果 print 出來
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, num_epoch, time.time()-epoch_start_time, \
      train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))
    train_loss_list1.append(train_loss/train_val_set.__len__())
    train_acc_list1.append(train_acc/train_val_set.__len__())

plt.plot(np.arange(num_epoch),train_loss_list1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(np.arange(num_epoch),train_acc_list1)
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.show()

test_set = ImgDataset(test_x)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        data = data.type(torch.FloatTensor)
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y,))

torch.save(model_best,'model.pt')