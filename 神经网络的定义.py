import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

X_numpy,Y_numpy = datasets.make_regression(
    n_samples=100,#样本数
    n_features=1,#特征数，即包含几个维度
    noise=10,#20%的数据产生偏离
    random_state=12 #随机数列，保持每次不变
)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y=Y.view(100,1)
# 线性函数模型的定义和训练:
# 1.定义模型 (包括正向传播方法 forward)
n_sample,in_features= X.shape
n_labels,out_features =Y.shape

class MyModle(torch.nn.Module):
    #初始化函数的定义，定于神经网络和参数
    def __init__(self,in_features_len,out_features_len):
        super(MyModle,self).__init__()

        self.linear = torch.nn.Linear(
            in_features_len,
            out_features_len
        )
    def forward(self,x):
        '''重写父类forward函数，正向传播'''
        out = self.linear(x)
        return out

model = MyModle(in_features,out_features)

# 2.定义损失(代价)函数loss
lossF = torch.nn.MSELoss()
opitimizer= torch.optim.SGD(model.parameters(),lr = 0.1)
print(list(model.parameters()))

n_iters= 100

for epoch in range(n_iters):
# 1.正向传播
    pre = model(X)
#2.计算损失
    ls = lossF(Y,pre)
# 3.向后传播 (计算梯度)
    ls.backward()
#4.更新权重，即向梯度反方向走一步，由优化器完成
    opitimizer.step()
#5.清空梯度，由优化器完成
    opitimizer.zero_grad()
    if epoch % 10 == 0:
        w,b = model.parameters()
        print(f"loss={ls.item():.8f},w={w.item():.4f},b={b.item():.4f}")


with torch.no_grad():
    X_test = torch.tensor(X_numpy, dtype=torch.float32)
    predicted = model(X_test).numpy()

plt.plot(X_numpy, Y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show()
