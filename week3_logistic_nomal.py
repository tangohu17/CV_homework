## Logistic Regression
###############################
import numpy as np
import random
import math

def inference(w, x):
    pred_y = 1/(1+math.exp(-w*x))
    return pred_y

def eval_loss(w, x_list, gt_y_list):
    avg_loss = 0.0
    for i in range(len(x_list)):
        avg_loss += (gt_y_list[i]*math.log10(1/(1+math.exp(-w*x_list[i]))) + (1-gt_y_list[i])*math.log10(1-(1/(1+math.exp(-w*x_list[i])))))   # loss function
    avg_loss /= -len(gt_y_list)
    return avg_loss

def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = diff * x
    return dw

def cal_step_gradient(batch_x_list, batch_gt_y_list, w, lr):
    avg_dw = 0
    batch_size = len(batch_x_list)
    #print(bat)
    for i in range(batch_size):
        pred_y = inference(w, batch_x_list[i])	# get label data
        dw = gradient(pred_y, batch_gt_y_list[i], batch_x_list[i])
        avg_dw += dw
    avg_dw /= batch_size
    w -= lr * avg_dw
    return w

def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    num_samples = len(x_list)
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w = cal_step_gradient(batch_x, batch_y, w, lr)
        print('w:{0}'.format(w))
        print('loss is {0}'.format(eval_loss(w, x_list, gt_y_list)))

def gen_sample_data():
    w = random.randint(0, 100) + random.random()		# for noise random.random[0, 1)
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 10) * random.random()
        y = 1/(1+math.exp(-w*x)) + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    return x_list, y_list, w

def run():
    x_list, y_list, w = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_list, y_list, 50, lr, max_iter)

if __name__ == '__main__':
    run()
