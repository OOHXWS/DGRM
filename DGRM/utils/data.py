import pandas as pd
import numpy as np
import torch



def read_ml100k(train_path, test_path, sep, header):
    train_set_dict, test_set_dict, item_set_dict = {}, {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)-1
    df_test = pd.read_csv(test_path, sep=sep, header=header)-1
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id, score = item[1], item[2], item[3]+1
        train_set_dict.setdefault(uid, {}).setdefault(i_id, score)
        item_set_dict.setdefault(i_id, {}).setdefault(uid, score)
    for item in df_test.itertuples():
        uid, i_id, score = item[1], item[2], item[3]+1
        test_set_dict.setdefault(uid, {}).setdefault(i_id, score)
    return train_set_dict, test_set_dict, item_set_dict



def read_ml1m(train_path, test_path, sep, header):  #train_path, test_path, sep, header
    # train_set_dict, test_set_dict = {}, {}
    # train_set = torch.load(train_path)
    # test_set = torch.load(test_path)
    # for uid, iid, score in train_set:
    #     train_set_dict.setdefault(uid, {}).setdefault(iid, score)
    # for uid, iid, score in test_set:
    #     test_set_dict.setdefault(uid, {}).setdefault(iid, score)
    # return train_set_dict, test_set_dict

    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)-1
    df_test = pd.read_csv(test_path, sep=sep, header=header)-1
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id, score = item[1], item[2], item[3]+1
        train_set_dict.setdefault(uid, {}).setdefault(i_id, score)
    for item in df_test.itertuples():
        uid, i_id, score = item[1], item[2], item[3]+1
        test_set_dict.setdefault(uid, {}).setdefault(i_id, score)
    return train_set_dict, test_set_dict


def read_ciao(train_path, test_path, sep, header):
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)-1
    df_test = pd.read_csv(test_path, sep=sep, header=header)-1
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id, score = item[1], item[2], item[3]+1
        train_set_dict.setdefault(uid, {}).setdefault(i_id, score)
    for item in df_test.itertuples():
        uid, i_id, score = item[1], item[2], item[3]+1
        test_set_dict.setdefault(uid, {}).setdefault(i_id, score)
    return train_set_dict, test_set_dict


def read_yh(train_path, test_path, sep, header):
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header) - 1
    df_test = pd.read_csv(test_path, sep=sep, header=header) - 1
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id, score = item[1], item[2], item[3] + 1
        train_set_dict.setdefault(uid, {}).setdefault(i_id, score)
    for item in df_test.itertuples():
        uid, i_id, score = item[1], item[2], item[3] + 1
        test_set_dict.setdefault(uid, {}).setdefault(i_id, score)
    return train_set_dict, test_set_dict


def get_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set, test_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    train_score_matrix, test_score_matrix = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    train_mask_matrix, test_mask_matrix = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))

    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            score = train_set_dict[u][i]
            train_set[u][i] = 1
            train_score_matrix[u][i] = score
            train_mask_matrix[u][i] = 1
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            score = test_set_dict[u][i]
            test_set[u][i] = 1
            test_score_matrix[u][i] = score
            test_mask_matrix[u][i] = 1
    return train_set, test_set, \
           train_score_matrix, test_score_matrix, \
           train_mask_matrix, test_mask_matrix
