# pytorch 进行梯度下降
import torch

m1 = torch.randn(1,requires_grad=True)
m2 = torch.randn(1,requires_grad=False)
b1 = torch.randn(1)
b2 = torch.randn(1)

print(m1)
print(m2)

print(b1,b2)

def forward1(x):
    global m1,b1
    return m1 * x + b1
def forward2(x):
    global m2,b2
    return m2 * x + b2
data  = [ [2,5] ] # m = 2 ,b =1 最佳
x = data[0][0]
y = data[0][1]

# 1.前向传播,构建了计算图
predipt1 = forward1(x)
predict2 = forward2(x)

# 构造损失(代价)函数
loss1 = (y-predipt1)**2
loss2 = (y -predict2)**2

# 2.向后传播
# 向后传播时自动计算梯度
loss1.backward()
loss2.backward()
print(loss1.grad_fn)
print(loss2.grad_fn)
print(m1.grad)
print(m2.grad)


