import csv
from torch.utils.data import DataLoader
from torch import nn
import torch
from numpy import *
import os
from sentence_transformers import SentenceTransformer,  InputExample,models,util
from sklearn.metrics import f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

m = SentenceTransformer('all-MiniLM-L6-v2')


sts_dataset_path_1 = 'train_720.csv'
sts_dataset_path_2 = 'test_720.csv'

train_samples = []
with open(sts_dataset_path_1, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn)
    for row in reader:
        score = torch.tensor(int(row['label']))
        s=row['sentence']
        e=m.encode(s,convert_to_tensor=True)
        train_samples.append([e,score])

test_samples = []
with open(sts_dataset_path_2, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn)
    for row in reader:
        score = torch.tensor(int(row['label']))
        s=row['sentence']
        e=m.encode(s,convert_to_tensor=True)
        test_samples.append([e,score])

en=[i[0]  for i in train_samples]
en=torch.stack(en,0)

l=[i[1]  for i in train_samples]
l=torch.stack(l,0)

test_en=[i[0]  for i in test_samples]
test_en=torch.stack(test_en,0)

test_l=[i[1]  for i in test_samples]
test_l=torch.stack(test_l,0)

train_loader = DataLoader(dataset=train_samples, batch_size=25, shuffle=True, num_workers=0, drop_last=False)

class MLP(nn.Module):
    def __init__(self):#初始化网络结构
        super(MLP,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Linear(10, 2))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,x): #搭建用两层全连接组成的网络模型
        # print("x.shape:",x.shape)
        out=self.linear(x)
        return out

    def predict(self,x):#实现LogicNet类的预测接口
        #调用自身网络模型，并对结果进行softmax处理,分别得出预测数据属于每一类的概率
        pred = torch.softmax(self.forward(x),dim=1)
        return torch.argmax(pred,dim=1)  #返回每组预测概率中最大的索引

    def getloss(self,x,y): #实现LogicNet类的损失值计算接口
        y_pred = self.forward(x)
        loss = self.criterion(y_pred,y)#计算损失值得交叉熵
        return loss


model=MLP()

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

model.apply(init_weights)


#loss = nn.CrossEntropyLoss(reduction='none').cuda()
trainer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    losses = 0.0  # 定义列表，用于接收每一步的损失值
    for X, y in train_loader:
        loss = model.getloss(X, y)
        trainer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 反向传播损失值
        trainer.step()  # 更新参数
        losses +=loss.item() * X.size(0)
    print("epoch:{}\ttrain loss:{:6f}".format(epoch+1,losses))


an=model.predict(test_en)

f1 = f1_score(an, test_l)
print("score:",f1)  #0.57 0.59

