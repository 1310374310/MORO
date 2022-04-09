from numpy.core.shape_base import stack
from metrics import *
from Params import args
import torch
import numpy as np
from time import time
import heapq
import multiprocessing

cores = multiprocessing.cpu_count() // 2
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

BATCH_SIZE = args.batch
batch_test_flag = args.batch_test_flag


'''heap rank'''
def ranklist_by_heapq(user_pos_test,test_items,rating,Ks):
    item_score = {}
    for i in range(len(test_items)):
        item_score[test_items[i]] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [],[],[],[]

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))
    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    # all_items = set(range(0, n_items))
    # test_items = list(all_items - set(training_items))
    test_items = x[2]

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test(model, datahandler, args):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = args.item
    n_users = args.user

    global train_user_set, test_user_set
    train_user_set = datahandler.train_user_set
    test_user_set =  datahandler.test_user_set

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE

    # test_users = list(test_user_set.keys())

    test_users = []
    for i in test_user_set.keys():
        if len(test_user_set[i])!=0:
            test_users.append(i)

    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]

        test_batch = datahandler.sampleTestBatch(start,end)
        pos_idx = test_batch['pos_items'] 
        neg_idx = test_batch['neg_items'].view(-1,99)
        pos_scores,neg_scores = model(test_batch,test_batch['users'],pos_idx,flag='Test')
        idex_batch = torch.cat([pos_idx.unsqueeze(1),neg_idx],dim=1).detach().cpu()
        rate_batch = torch.cat([pos_scores.unsqueeze(0),neg_scores],dim=0).permute(1,0).detach().cpu()
        #batch_result = []
        # for i in range(u_batch_size):
        #     batch_result.append(test_one_user((rate_batch[i],user_list_batch[i],idex_batch[i])))
        user_batch_rating_uid = zip(rate_batch, user_list_batch, idex_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

    assert count == n_test_users 
    pool.close()
    return result


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop