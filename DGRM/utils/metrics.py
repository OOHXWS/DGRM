import numpy  as np


def precision(rec_list, target_list):
    rec_list = set(rec_list)
    target_list = set(target_list)
    nb_common = len(rec_list&target_list)
    return nb_common/len(rec_list)


def recall(rec_list, target_list):
    rec_list = set(rec_list)
    target_list = set(target_list)
    nb_common = len(rec_list&target_list)
    return nb_common/len(target_list)


def f1(precision_value, recall_value):
    return 2*precision_value*recall_value/(precision_value+recall_value)


def hr(rec_list, target_list):
    rec_list = set(rec_list)
    target_list = set(target_list)
    nb_common = len(rec_list & target_list)
    nb_target = len(target_list)
    return nb_common, nb_target


def dcg(rec_list, target_list):
    '''
    但对于单个用户的dcg
    :param rec_list:
    :param target_list:
    :return:
    '''
    result = 0
    for i, item in enumerate(rec_list):
        if item in target_list:
            # 因为i从0开始，因此要多加一个1
            result += 1/np.log2(i+1+1)
    fenmu = sum([1/np.log2(i+1+1) for i in range(len(rec_list))])
    result = result/fenmu
    return result


def total_k(topk, predict_matrix, mask_matrix, target_uis_dict):
    '''
    得到对应的
    :param topk:取前几个物品
    :param target_matrix: 目标的矩阵
    :param mask_matrix: 就是训练集中的矩阵
    :param target_uis_dict: 测试集中的评分字典
    :return: precision, recall, f1_score, hr
    '''
    predict_matrix = predict_matrix - 999*mask_matrix
    users = target_uis_dict.keys()
    # 记录被推荐项目的种类
    items_rec_set = set()
    # hr指标的统计
    nb1_hr, nb2_hr = 0, 0
    # 计算ndcg的指标
    dcg_list = []
    # 计算总的precision和recall相关的记录
    precision_list, recall_list = [], []
    for u in users:
        # 得到用户u的评分向量
        score_vec = predict_matrix[u]
        # 创建score和user_id的关系
        score_idx_list = [(score, idx) for idx, score in enumerate(score_vec)]
        # 根据score降序排序
        score_idx_list.sort(key=lambda x:x[0], reverse=True)
        rec_list = [idx for _, idx in score_idx_list][:topk]
        target_list = target_uis_dict[u].keys()
        # precision和recall相关的
        value_precision = precision(rec_list=rec_list, target_list=target_list)
        value_recall = recall(rec_list=rec_list, target_list=target_list)
        precision_list.append(value_precision)
        recall_list.append(value_recall)
        # # hr指标相关的
        # tmp1, tmp2 = hr(rec_list=rec_list, target_list=target_list)
        # nb1_hr += tmp1
        # nb2_hr += tmp2
        # coverage指标相关
        items_rec_set = items_rec_set | set(rec_list)
        # ndcg指标相关
        value_dcg = dcg(rec_list, target_list)
        dcg_list.append(value_dcg)
    v_precision, v_recall = np.mean(precision_list), np.mean(recall_list)
    v_f1_score = f1(v_precision, v_recall)
    # # v_hr = nb1_hr/nb2_hr
    # v_hr = nb1_hr / (len(users)*topk)
    v_ndcg = np.mean(dcg_list)
    nb_converage = len(items_rec_set)
    converage = nb_converage/mask_matrix.shape[1]
    return v_precision, v_recall, v_f1_score, v_ndcg, converage