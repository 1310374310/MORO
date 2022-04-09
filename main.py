from Model.MORO import MORO
from DataHandler import DataHandler
from prettytable import PrettyTable
from Params import args
import torch
import numpy as np
import random
import logging
import time
import torch.nn as nn

from evaluate import test, early_stopping

time_stamp = int(time.time())
time_array = time.localtime(time_stamp)
str_date = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
formatter=logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('./log/'+str_date+'_'+args.data+'.log',)
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)



def main():
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    device = torch.device('cuda:'+str(args.gpu_id) if args.cuda else 'cpu')

    # load data
    datahandler = DataHandler()
    datahandler.LoadData() 
    logger.info("dataset: "+ str(args.data))
    logger.info("n_users: " + str(args.user))
    logger.info("n_items: " + str(args.item))
    logger.info("n_behaviors: " + str(args.behNum))
    logger.info("n_interactions: " + str(args.interactions))
    logger.info(args)
    

    # init model
    if args.weighted:
        model = MORO(args,datahandler.trainMats_W)
    else:
        model = MORO(args,datahandler.trainMats)
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info(model)
    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    # process_batch
    logger.info("start training ...")
    for epoch in range(args.epoch):
        if args.shuffle==True:
            index = np.arange(datahandler.users.shape[0])
            np.random.shuffle(index)
            datahandler.users = datahandler.users[index]
            users = datahandler.users[:10000]

        model.train()
        loss,s = 0,0
        train_s_t  =time.time()
        
        while s + args.batch <= len(users):
            batIds =  users[s:s+args.batch]     
            batch = datahandler.sampleTrainBatch(batIds,neg_nums=10)
            temp = [i for i in batch['pos_items']]
            batIIds = torch.cat(temp,dim=0).unique()
            pos_scores,neg_scores, reg , w = model(batch,batIds,batIIds,flag='Train')
            batch_loss = model.cal_loss_L(pos_scores,neg_scores)
            batch_loss += reg
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
            s += args.batch
           
        train_e_t = time.time()

        '''testing'''
        if epoch%10==9 or epoch==1:
            model.eval()
            test_s_t = time.time()
            ret = test(model, datahandler,args)
            test_e_t = time.time()


            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio","auc"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio'], ret['auc']]
            )
            logger.info(train_res)
        
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['hit_ratio'][-1],
                                                                        cur_best_pre_0,
                                                                        stopping_step, 
                                                                        expected_order='acc',
                                                                        flag_step=10)

            if should_stop:
                break
    
            """save weight"""
            if ret['hit_ratio'][-1] == cur_best_pre_0 and args.save:
                torch.save(model.state_dict(), args.out_dir + 'model_' + args.data + '.ckpt')
                
        else:
            logger.info('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    logger.info('early stopping at %d, hit_ratio@10:%.4f' % (epoch, cur_best_pre_0))

if __name__=='__main__':
    main()









