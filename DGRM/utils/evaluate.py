import copy
import random

import numpy as np


def recall_precision_f1_hr_ndcg(predict_matrix,liability_threshold,liability_matrix, train_mask_matrix,test_mask_matrix,
                           test_uis_dict,top_k):
    '''
    计算方法名上的数据
    :param predict_matrix: np.array 对每个用户的每个项目都预测评分，这是一个矩阵
    :param train_mask_matrix: 隐藏掉已经购买过的矩阵
    :param test_uis_dict: 字典
    :param top_k: 推荐多少个物品
    :return: recall, precision, f1-score, hr
    '''
    recalls, precisions, ndcgs = [], [], []
    total_hit, total_rec = 0, 0
    users = list(test_uis_dict.keys())
    predict_matrix = predict_matrix-99*train_mask_matrix
    p = 0
    for u in users:
        items_of_u_all = [it_o_u for it_o_u, rating in test_uis_dict[u].items()]
        items_of_u = [it_o_u for it_o_u, rating in test_uis_dict[u].items() if rating >= 3]
        items_of_u_list = sorted(items_of_u, key=lambda x: test_uis_dict[u][x], reverse=True)[:top_k]
        score_vec = predict_matrix[p]
        p+=1
        score_idx_list = [idx for idx in items_of_u_all if score_vec[idx] >= np.mean(score_vec)]
        score_idx_list = sorted(score_idx_list, key=lambda x: score_vec[x], reverse=True)[:top_k]
        hit_order = np.isin(score_idx_list, items_of_u_list).astype(int)
        hit_order = np.append(hit_order, np.zeros(5 - len(hit_order)))
        hit = np.sum(hit_order)
        total_hit += hit
        total_rec += top_k
        if hit == 0:
            recalls.append(0)
            precisions.append(0)
            ndcgs.append(0)
        else:
            recall = hit / len(items_of_u_list)
            recalls.append(recall)
            precision = hit / top_k
            precisions.append(precision)
            idx_arr = np.arange(1, len(hit_order) + 1, 1)
            ideal_arr = 1 / np.log2(idx_arr + 1)
            idcg = np.sum(ideal_arr)
            dcg_arr = np.multiply(ideal_arr, hit_order)
            dcg = np.sum(dcg_arr)
            ndcg = dcg / idcg
            ndcgs.append(ndcg)

    return top_k, recalls, precisions, ndcgs, total_hit, total_rec