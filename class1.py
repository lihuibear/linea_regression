import  numpy as np
import matplotlib.pyplot as plt
import itertools

data = np.array(
    [
        [80,200],
        [95,230],
        [104,245],
        [112,247],
        [125,259],
        [135,262]
    ]
)

X = data[:,0]
Y = data[:,1]
# 绘制图形
plt.scatter(X,Y,c="red")
plt.show()

com_lists = list(itertools.combinations(data,2))
# 强制转换为list
print(com_lists)
print(len(com_lists))
# 建立两个容器，存m,b;
ms = []
bs = []
for comlist in com_lists:
    x1,y1 = comlist[0]
    x2,y2 = comlist[1]
    # 因有下列等式成立
    # y1 = m * x1 + b
    # y2 = m * x2 + b
    #所以
    m = (y2-y1) / (x2-x1)
    b = y1 - m * x1
    # or b = y2 - m * x2
    ms.append(m)
    bs.append(b)
m,b = np.mean(ms),np.mean(bs)
print(ms,bs)
print(m,b)

print("=================")

x = 140
predict_fx =m * x + b
print(f"140:{predict_fx}")


losses = []
for x,y in data:
    predict = m * x + b
    loss = (y-predict)**2
    losses.append(loss)
print(losses)
print(np.mean(losses))
# 平均损失值：70.89464479297973


print("+++++++++++++++")

