import pickle
from scipy import sparse

from torch.nn.init import zeros_
from Params import args
import numpy as np
from scipy.sparse import coo_matrix
import torch
device = torch.device('cuda:'+str(args.gpu_id) if args.cuda else 'cpu')

class DataHandler:
    
    def __init__(self):
        if args.data == 'Tmall':
            predir = './Datasets/Tmall/'
            behaviors = ['pv', 'fav', 'cart', 'buy']
            #behaviors = [ 'buy']
        elif args.data == 'beibei':
            predir = './Datasets/beibei/'
            behaviors = ['pv', 'cart', 'buy']
            #behaviors = ['buy']
        elif args.data == 'yelp':
            predir = 'Datasets/yelp/buy/'
            behaviors = ['tip', 'neg', 'neutral', 'pos']
            # behaviors = [ 'tip', 'neg', 'pos']
            #behaviors = ['pos']
        elif args.data == 'ijcai':
            predir = './Datasets/ijcai/'
            behaviors = ['click', 'fav', 'cart', 'buy']
        elif args.data == 'ml10m':
            predir = 'Datasets/MultiInt-ML10M/buy/'
            behaviors = ['neg', 'neutral', 'pos']

        self.predir = predir
        self.behaviors = behaviors  
        self.train_file = predir + 'trn_'
        self.test_file = predir + 'tst_'


    def LoadData(self):
        trainMats = list()
        Mats = list()
        total_mat = 0
        col = list()
        row = list()
        values = list()

        for i in range(len(self.behaviors)):
            behaviors = self.behaviors[i]
            path = self.train_file + behaviors
            with open(path, 'rb') as fs:
                mat = (pickle.load(fs) != 0).astype(np.float32)
                #total_mat = np.maximum(mat.A,total_mat)
            trainMats.append(self.trans2coo(mat))
            col +=list(self.trans2coo(mat).col)
            row +=list(self.trans2coo(mat).row)
            values+=list(self.trans2coo(mat).data)
            Mats.append(mat)
        temp = set([(row[i],col[i],values[i]) for i in range(len(col))])
        row = [i[0] for i in temp]
        col = [i[1] for i in temp]
        values = [i[2] for i in temp]
        total_mat = sparse.coo_matrix((values,(row,col)),shape=trainMats[0].shape)
        trainMats.insert(0,self.trans2coo(total_mat))
        Mats.insert(0,self.trans2coo(total_mat))

        

        # test set
        path = self.test_file +'int'
        with open(path, 'rb') as fs:
            testInt = np.array(pickle.load(fs))
        testStat = (testInt !=None)
        testUsers = np.reshape(np.argwhere(testStat != False), [-1])
        
        self.trainMats = trainMats
        self.testInt = testInt
        self.testUser = testUsers
        self.Mats = Mats

        # user, # item and # behaviors
        args.user, args.item = self.trainMats[0].shape
        args.interactions = 0
        for mat in self.trainMats[1:]:
            args.interactions+= np.sum(np.sum(mat,axis=-1)).astype(np.int)
        args.behNum = len(trainMats)

        self.prepareData()
        
        

    def prepareData(self,beh=-1):
        targetMat = self.trans2coo(self.trainMats[beh])
        train_cf = []
        for i in range(targetMat.nnz):
            train_cf.append([targetMat.row[i],targetMat.col[i]])
        
        test_cf = []
        test_u = np.where(self.testInt!=None)   
        for u in test_u[0]:
            test_cf.append([u,self.testInt[u]]) 
        self.train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
        self.test_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))
        
        self.train_user_set = {}
        for i in train_cf: 
            if i[0] not in self.train_user_set.keys():
                self.train_user_set[i[0]]=[i[1]]
            else:
                self.train_user_set[i[0]].append(i[1])

        # 用户的集合
        self.users = torch.LongTensor(np.array(range(args.user),np.int32))

        self.test_user_set = {}
        for i in test_cf: 
            if i[0] not in self.test_user_set.keys():
                self.test_user_set[i[0]]=[i[1]]
            else:
                self.test_user_set[i[0]].append(i[1])
        self.constraintMats=[]
        #self.get_constraintMats()
        self.get_trainMats_weighted()
             
        
    def sampleTrainBatch(self,batIds,neg_nums=99):
        def negSamp(temLabel, sampSize, nodeNum):
            negset = [None] * sampSize
            cur = 0
            while cur < sampSize:
                rdmItm = np.random.choice(nodeNum)
                if temLabel[rdmItm] == 0:
                    negset[cur] = rdmItm
                    cur += 1
            return negset

        flag = 0
        batch = len(batIds)
        res_users = []
        res_pos = []
        res_negs = []
        if flag ==0:
            temlen = batch*neg_nums*args.behNum
            for b in range(1,args.behNum):
                users = [None] * temlen
                pos = [None] * temlen
                negs = [None] * temlen
                cur = 0
                labelMat = self.Mats[b]
                temLabel = labelMat[batIds].toarray()
                for  i in range(batch):
                    posset = np.reshape(np.argwhere(temLabel[i]!=0),[-1])
                    sampNum = min(neg_nums, len(posset))
                    if sampNum == 0:
                        poslocs = []
                        neglocs = []
                    else:
                        poslocs =np.random.choice(posset,sampNum)
                        neglocs = negSamp(temLabel[i], sampNum, args.item)
                    
                    for j in range(sampNum):
                        posloc = poslocs[j]
                        negloc = neglocs[j]
                        users[cur] = int(batIds[i])
                        pos[cur] = int(posloc)
                        negs[cur] = int(negloc)
                        cur += 1
                users = users[:cur]
                pos  = pos[:cur]
                negs = negs[:cur]
                res_users.append(torch.LongTensor(np.array(users,np.int32)).to(device))
                res_pos.append(torch.LongTensor(np.array(pos,np.int32)).to(device))
                res_negs.append(torch.LongTensor(np.array(negs,np.int32)).to(device))
        else:
            temlen = batch*neg_nums
            users = [None] * temlen
            pos = [None] * temlen
            negs = [None] * temlen
            cur = 0
            labelMat = self.Mats[-1]
            temLabel = labelMat[batIds].toarray()
            for  i in range(batch):
                posset = np.reshape(np.argwhere(temLabel[i]!=0),[-1])
                sampNum = min(neg_nums, len(posset))
                if sampNum == 0:
                    poslocs = []
                    neglocs = []
                else:
                    poslocs =np.random.choice(posset,sampNum)
                    neglocs = negSamp(temLabel[i], sampNum, args.item)
                
                for j in range(sampNum):
                    posloc = poslocs[j]
                    negloc = neglocs[j]
                    users[cur] = int(batIds[i])
                    pos[cur] = int(posloc)
                    negs[cur] = int(negloc)
                    cur += 1
                users = users[:cur]
                pos  = pos[:cur]
                negs = negs[:cur]
                res_users.append()

        
        batch = {}
        batch['users']= res_users
        batch['pos_items'] = res_pos
        batch['neg_items'] = res_negs
        return batch


    def sampleTestBatch(self,start,end,neg_nums=99):
        def negative_sampling(user_item,user_set,neg_nums):
            neg_items=[]
            for user, _ in user_item.cpu().numpy():
                user = int(user)
                count = 0
                while count<neg_nums:
                    neg_item = np.random.randint(low=0, high=args.item, size=neg_nums*2)
                    for item in neg_item:
                        if item not in user_set[user]:
                            neg_items.append(item)
                            count+=1
                        if count==neg_nums:
                            break
            return neg_items
        
        batch = {}
        batch_cf = self.test_cf[start:end]
        batch['users']= batch_cf[:,0]
        batch['pos_items'] = batch_cf[:,1]
        batch['neg_items'] = torch.LongTensor(negative_sampling(batch_cf,self.test_user_set,neg_nums))
        return batch

        

    def trans2coo(self,Mat):
        Mat = coo_matrix(Mat)
        return Mat
    
    def get_constraintMats(self):
        self.constraintMats=[]
        for train_mat in self.trainMats:
            items_D = np.sum(train_mat, axis = 0).reshape(-1)
            users_D = np.sum(train_mat, axis = 1).reshape(-1)

            beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
            beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

            constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))
            constraint_mat = constraint_mat.flatten()
            self.constraintMats.append(constraint_mat)

    def get_trainMats_weighted(self):
        self.trainMats_W = []
        for train_mat in self.trainMats:
            items_D = np.sum(train_mat, axis = 0).reshape(-1)
            users_D = np.sum(train_mat, axis = 1).reshape(-1)

            beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
            beta_iD = (1 / np.sqrt(items_D + 1)).reshape(-1, 1)

            row = train_mat.row
            col = train_mat.col
            values = np.multiply(beta_uD[row],beta_iD[col]).A.flatten()
            mat = coo_matrix((values,(row,col)),shape=(args.user,args.item)) 
            #mat = (beta_uD.dot(beta_iD).A)
            self.trainMats_W.append(mat)
        return 


if __name__=='__main__':
    datahandler = DataHandler()
    datahandler.LoadData()



