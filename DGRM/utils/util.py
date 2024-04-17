import numpy as np
import random

import torch


def neg_sample(train_vec, nb_sample):
    data = np.array(train_vec)
    idx = np.zeros_like(data)
    for i in range(data.shape[0]):
        # 得到所有为0的项目下标
        items = np.where(data[i] == 0)[0].tolist()
        # 随机抽取一定数量的下标
        tmp_zr = random.sample(items, nb_sample)
        # 这些位置的值为1
        idx[i][tmp_zr] = 1
    return idx


def regularization(weights, alpha):
    '''正则化的操作'''
    item = sum([torch.norm(w, p=2) for w in weights])
    item = alpha*item
    return item


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class EarlyStopper:
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_metric = 0
        self.save_path = save_path

    def is_continuable(self, model, metric):  #如果num_trials次都没有更好的指标出现则停止训练
        #torch.save(model, self.save_path)
        if metric > self.best_metric:
            self.best_metric = metric
            self.trial_counter = 0
            torch.save(model, self.save_path)
            print('successfully saved')

            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class EarlyStopper_liability:
    def __init__(self, num_trials, save_path_G):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_metric = 0
        self.save_path_G = save_path_G


    def is_continuable(self, model_G , metric):#如果num_trials次都没有更好的指标出现则停止训练
        torch.save(model_G, self.save_path_G)
        print("successfully saved")
        if metric > self.best_metric:
            self.best_metric = metric
            self.trial_counter = 0
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
