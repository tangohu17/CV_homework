““loss值减到一定值就不在减小，这个值不是很小，有时是个位数甚至更大，什么原因造成的？””

# Linear Regression
import numpy as np
import random

def eval_loss(X, Y, theta=[[0],[0]]):
    """

    :param X:
    :param y: y是一维的行向量（对应np的一列）
    :param theta: theta是一维列向量，是x0x1的参数初始值
    :return: avg_loss是损失函数。np.square（）计算数组各元素的平方
    """
    avg_loss = 0.0
    m = len(Y)
    pred_y = X.dot(theta)
    avg_loss = 1.0 / (2 * m) * (np.sum(np.square(pred_y - Y)))
    return  avg_loss

def gradientDescent(X, Y, theta=[[0],[0]], alpha=0.01, max_iters=500):
    """

    :param X:
    :param y: y是一维的行向量
    :param theta: theta是一维列向量
    :param alpha: 步长
    :param max_iters: 迭代次数
    :return:
    """
    m = len(Y)
    # 定义一个500维的一维列向量，值全是0
    avg_loss_history = np.zeros(max_iters)
    #  进行500次迭代
    for i in np.arange(max_iters):
        pred_y = X.dot(theta)
        # 梯度下降方向为avg_loss函数对参数求导数为(1.0 / m) * (X.T.dot(h - y))
        theta = theta - alpha*(1.0 / m)*(X.T.dot(pred_y - Y))
        avg_loss_history[i] = eval_loss(X, Y, theta)
        print('loss is {0}'.format(avg_loss_history[i]))
    return (theta, avg_loss_history)

if __name__ == '__main__':
    X = np.random.rand(1000).reshape(500, 2)
    Y = np.random.rand(500)
    gradientDescent(X, Y, max_iters=500)

