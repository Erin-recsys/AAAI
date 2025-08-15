import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import datetime

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    import time

    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.Loss = loss_class
    
    with timer(name="Sample"):
    #    S = utils.UniformSample_original_python(dataset)
        if utils.sample_ext:
            S = utils.UniformSample_original(dataset)  
        else:
            S = utils.UniformSample_original_python(dataset)  
    
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1

    batch_times = []
    aver_loss = 0.
    
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):



        batch_start_time = time.time()
        bpr_loss= bpr.stageOne(batch_users, batch_pos, batch_neg)

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        aver_loss += bpr_loss
        
        
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', bpr_loss, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

            
    aver_loss = aver_loss / total_batch


    avg_batch_time = np.mean(batch_times)
    total_training_time = sum(batch_times)
    

    time_info = timer.dict()
    timer.zero()
    #time_info = timer.dict()
    #timer.zero()
    
    
    #return f"loss{aver_loss:.3f}-delta{aver_delta_loss:.3f}-adv{aver_adv_loss:.3f}-cat{aver_category_loss:.3f}-{time_info}"
    return f"loss{aver_loss:.3f}-{time_info}-avg_batch:{avg_batch_time:.4f}s"
def test_one_batch(X, dataset, users):
    sorted_items = X[0].numpy()  
    groundTrue = X[1]  
    r = utils.getLabel(groundTrue, sorted_items)
    recall, hit_ratio, entropy_vals = [], [], []
    coverage = []  
    

    for k_idx, k in enumerate(world.topks):

        user_coverage = []
        

        for user_idx, items in enumerate(sorted_items):
            user_items = items[:k]
            covered_genres = set()
            for item in user_items:
                if item in dataset.game_to_genres:  
                    covered_genres.update(dataset.game_to_genres[item])
            
            user_coverage.append(len(covered_genres))
        
        coverage.append(np.mean(user_coverage))

        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        recall.append(ret['recall'])

        hit_ret = utils.HitRatio_ATk(groundTrue, r, k)
        hit_ratio.append(hit_ret['hit_ratio'])

        entropy_sum = utils.Entropy_ATk(sorted_items, dataset, k)
        entropy_vals.append(entropy_sum)

    return {
        'recall': np.array(recall), 
        'hit_ratio': np.array(hit_ratio),
        'coverage': np.array(coverage),
        'entropy': np.array(entropy_vals)
    }

def Test(dataset, Recmodel, epoch, w=None, multicore=0):

    u_batch_size = world.config['test_u_batch_size']
    testDict = dataset.testDict
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)

    results = {
        'recall': np.zeros(len(world.topks)),
        'hit_ratio': np.zeros(len(world.topks)),
        'coverage': np.zeros(len(world.topks)),
        'entropy': np.zeros(len(world.topks))
    }
    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(lambda x: test_one_batch(x, dataset), X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, dataset, batch_users))
                
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['hit_ratio'] += result['hit_ratio']
            results['coverage'] += result['coverage']
            results['entropy'] += result['entropy']

        results['recall'] /= float(len(users))
        results['entropy'] /= float(len(users))
        if len(pre_results) > 0:
            results['hit_ratio'] /= float(len(pre_results))
            results['coverage'] /= float(len(pre_results))
        else:
            results['hit_ratio'] = np.zeros(len(world.topks))
            results['coverage'] = np.zeros(len(world.topks))
            
        if multicore == 1:
            pool.close()

    print(results)
    return results



