import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MyData
from layers_kl import Gen, Dis
# from matplotlib import pyplot as plt
import argparse
from utils.data import read_ml100k, get_matrix, read_ml1m, read_ciao, read_yh
from utils.evaluate import recall_precision_f1_hr_ndcg
from utils.util import EarlyStopper, setup_seed
import gaussian_diffusion_gan as gd



def TT(dataset, reliability_matrix, model, purchase_matrix, mask_matrix, test_uis_dict, tt_test_mask_matrix, topk=5):
    mean_type = 'x0'
    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max,
                                     args.steps + 1, device='cpu')
    dataloader_tt = DataLoader(dataset=dataset, shuffle=False, batch_size=128, drop_last=False)
    model.eval()

    total_hit, total_rec = 0, 0
    recalls, precisions, ndcgs = [], [], []
    reslutt = []
    with torch.no_grad():
        for idxs in dataloader_tt:
            test_uis_dict_u = {}
            idxs = list(idxs.detach().numpy())
            for u in idxs:
                test_uis_dict_u[u] = test_uis_dict[u]
            v = 0
            binary_matrix = torch.tensor(purchase_matrix[idxs], dtype=torch.float)
            z = torch.randn_like(binary_matrix)
            x = diffusion.q_sample(binary_matrix, diffusion.sample_timesteps(int(5), len(z)))
            mask_matrixx = torch.tensor(reliability_matrix[idxs], dtype=torch.float32)
            x_new = x
            for i in reversed(range(2)):
                t = torch.full((len(idxs),), i, dtype=torch.int64)
                if v == 0:
                    x_0 = (model(x, z, t)*4+1) * mask_matrixx
                else:
                    x_0 = (model(x, z, t)*4+1)
                x_new = diffusion.sample_posterior(x_0, x_new, t)
                x = x_new.detach()
                v += 1
            out = x.numpy()
            top_k, recall_u, precisions_u, ndcg_u, hit, rec = recall_precision_f1_hr_ndcg(out, 0, out,
                                                                                          mask_matrix[idxs],
                                                                                          tt_test_mask_matrix,
                                                                                          test_uis_dict_u, top_k=topk)

            recalls.extend(recall_u)
            precisions.extend(precisions_u)
            ndcgs.extend(ndcg_u)
            total_hit += hit
            total_rec += rec
    recall = sum(recalls) / len(recalls) if len(recalls) > 0 else 0
    precision = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
    hr = total_hit / total_rec if total_rec > 0 else 0
    f1_score = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0
    ndcg = sum(ndcgs) / len(ndcgs) if len(ndcgs) > 0 else 0

    return top_k, recall, precision, f1_score, hr, ndcg, reslutt



def train_DGRM(save_dir, purchase_matrix, interact_matrix, mask_matrix, train_set, test_uis_dict,
                tt_test_mask_matrix, nb_item, epoches, batch_size, alpha, top_k):
    save_path = os.path.join(save_dir, 'model_drgm_kl.pkl')
    early_stop = EarlyStopper(num_trials=128, save_path=save_path)  # when to stop
    dataset = MyData(nb_user)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    gen = Gen(nb_item)
    dis_ra = Dis(nb_item)
    loss_bce = torch.nn.BCELoss()
    d_real_label = torch.ones(batch_size, 1, dtype=torch.float)
    d_fake_label = torch.zeros(batch_size, 1, dtype=torch.float)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=0.0001, weight_decay=1e-2)
    dis_ra_opt = torch.optim.Adam(dis_ra.parameters(), lr=0.0002, weight_decay=1e-2)
    step_gen, step_dis = 2, 4
    mean_type = 'x0'
    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max,
                                     args.steps + 1, device='cpu')
    for e in range(epoches):
        # ------------------------------------------
        # Train D
        # ------------------------------------------
        dis_ra.train()
        gen.eval()
        for step in range(step_dis):
            for idxs in dataloader:
                x_start = torch.tensor(purchase_matrix[idxs], dtype=torch.float)
                eu = torch.tensor(interact_matrix[idxs], dtype=torch.float)
                t = diffusion.sample_timesteps(args.steps, batch_size)
                z = torch.randn_like(x_start)
                real_vec, real_vec_1 = diffusion.q_sample_pair(x_start, t)
                real_vec.requires_grad = True
                fake_x_start = gen(real_vec_1.detach(), z, t) * eu
                fake_vec = diffusion.sample_posterior(fake_x_start, real_vec_1, t)
                D_fake = dis_ra(real_vec_1.detach(), fake_vec, t)
                # D_real
                D_real = dis_ra(real_vec_1.detach(), real_vec, t)
                # KL
                loss_real = loss_bce(D_real, d_real_label[:len(D_real)])
                loss_fake = loss_bce(D_fake, d_fake_label[:len(D_fake)])
                loss_D = loss_real + loss_fake
                dis_ra_opt.zero_grad()
                loss_D.backward()
                dis_ra_opt.step()

        # ------------------------------------------
        # Train G
        # ------------------------------------------
        gen.train()
        dis_ra.eval()
        for step in range(step_gen):
            for idxs in dataloader:
                x_start = torch.tensor(purchase_matrix[idxs], dtype=torch.float)
                eu = torch.tensor(interact_matrix[idxs], dtype=torch.float)
                t = diffusion.sample_timesteps(args.steps, batch_size)
                z = torch.randn_like(x_start)
                real_vec, real_vec_1 = diffusion.q_sample_pair(x_start, t)
                fake_x_start = gen(real_vec_1.detach(), z, t) * eu
                fake_vec = diffusion.sample_posterior(fake_x_start, real_vec_1, t)
                D_fake = dis_ra(real_vec_1.detach(), fake_vec, t)
                loss_fake = loss_bce(D_fake, d_real_label[:len(D_fake)])
                loss_regular = alpha * torch.sum(
                    ((fake_x_start - x_start) * eu).pow(2))
                loss_G = loss_fake + loss_regular
                gen_opt.zero_grad()
                loss_G.backward()
                gen_opt.step()


        if (e + 1) % 2 == 0:
            top_k, recall, precision, f1_score, hr, ndcg, reslutt = TT(dataset, interact_matrix, gen, purchase_matrix, mask_matrix, test_uis_dict, tt_test_mask_matrix, topk=top_k)

            print('{}, recall:{}, precision:{}, f1:{}, hr:{}, ndcg:{} , loss_G:{}, loss_D:{}'
                  .format(e, recall, precision, f1_score, hr, ndcg, loss_G.item(), loss_D.item()))
            if not early_stop.is_continuable(gen, f1_score):
                print('Best：{}'.format(early_stop.best_metric))

                return


if __name__ == '__main__':
    setup_seed(10)
    parser = argparse.ArgumentParser()

    # params for diffusion
    parser.add_argument('--mean_type', type=str, default='eps', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.01, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=1, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=True, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=2, help='steps of the forward process during inference')

    parser.add_argument('--r1_gamma', type=float, default=0.01, help='coef for r1 reg')
    # 100k
    args = parser.parse_args()
    nb_user = 943
    nb_item = 1682
    train_set_dict, test_set_dict, item_set_dict = read_ml100k('dataset/ml-100k/trainset_100k.csv',
                                                               'dataset/ml-100k/testset_100k.csv', sep=',',
                                                               header=None)
    train_set, test_set, \
    train_score_matrix, test_score_matrix, \
    train_mask_matrix, test_mask_matrix = get_matrix(train_set_dict, test_set_dict, nb_user, nb_item)

    train_DGRM(save_dir='models/ml100k', purchase_matrix=train_score_matrix, interact_matrix=train_mask_matrix,
                mask_matrix=train_mask_matrix, train_set=train_set, test_uis_dict=test_set_dict,
                tt_test_mask_matrix=test_mask_matrix, nb_item=nb_item, epoches=50000, batch_size=256, alpha=0.1, top_k=5)

