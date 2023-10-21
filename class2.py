#梯度下降算法
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

# 初始化参数
m = 1
b = 1
lr  = 0.00001

# 梯度下降的函数
# 当前的m，b和数据 data，学习率lr
def gradientdecent(m,b,data,lr):
    loss,mpd,bpd = 0,0,0
    # loss 均方误差 ，mpd为m的偏导数，bpd为b的偏导数
    for xi,yi in data:
        loss += (m * xi + b - yi)**2
        bpd += (m * xi + b - yi) * 2
        mpd += (m * xi + b - yi) * 2 * xi

    #更新m,b
    N = len(data)
    loss = loss / N
    mpd = mpd / N
    bpd = bpd / N
    m = m - mpd * lr
    b = b - bpd * lr
    return loss,m,b


for ecoch in range(30000000):
    mse,m,b = gradientdecent(m,b,data,lr)
    if(ecoch % 100000) == 0:
        print(f"loss={mse:.4f},m={m:.4f},b={b:.4f}")

#  loss=42.8698,m=1.0859,b=122.6760