from torch._C import device
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class RGCNConv(nn.Module):
    def __init__(self, input_dim,output_dim,num_r):
        super(RGCNConv,self).__init__()
        self.num_r = num_r
        self.activate = nn.LeakyReLU()
        self.project = nn.Linear(input_dim,output_dim,bias=False)
        self.Ws = nn.ModuleList()
        for _ in range(self.num_r):
            self.Ws.append(nn.Linear(input_dim,output_dim,bias=False))
        

    def forward(self,u_emb,i_emb,mats):
        # 每种关系进行传播求和不做参数共享
        for i in range(self.num_r):
            div_ = 1/(torch._sparse_sum(mats[i],dim=1).to_dense().reshape(-1,1)+1)
            u_emb = self.project(u_emb) + torch._sparse_mm(mats[i],self.Ws[i](i_emb))*div_

            div_ = 1/(torch._sparse_sum(mats[i].t(),dim=1).to_dense().reshape(-1,1)+1)
            i_emb = self.project(i_emb) + torch._sparse_mm(mats[i].t(),self.Ws[i](u_emb))*div_

        u_emb = self.activate(u_emb)
        i_emb = self.activate(i_emb)
        
        return u_emb, i_emb


class MORO(nn.Module):
    def __init__(self,args,trainMats):
        super(MORO, self).__init__()
        self.n_item = args.item
        self.n_user = args.user
        self.n_behavior = args.behNum
        self.dim = args.dim
        self.n_hops = args.n_hops
        self.trainMats = []
        self.device = torch.device('cuda:'+str(args.gpu_id) if args.cuda else 'cpu')
        self.initial_weights()
        self.item_emb = nn.Parameter(self.item_emb)
        self.behavior_emb =  nn.Parameter(self.behavior_emb)
        self.user_emb =  nn.Parameter(self.user_emb)
        self.Rconv = RGCNConv(self.dim,self.dim,self.n_behavior-1)
        for mat in trainMats:
            self.trainMats.append(self._get_sparse_adj(mat))
        #self.constraintMats = constraintMats

        self.ls = args.ls
        self.FC1 = nn.Sequential(nn.Linear(self.dim*(self.n_behavior-1),self.dim))
        self.FC2 = nn.Linear(self.dim,self.dim)
        self.conbine = nn.Sequential(nn.Linear(self.dim,self.dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(self.dim,1))
        self.drop_out = nn.Dropout(0.2)
        self.activate = nn.LeakyReLU()
        self.layernorm = nn.LayerNorm(self.dim)
        self.projects = nn.ModuleList()
        for i in range(self.n_behavior-1):
            self.projects.append(nn.Linear(self.dim,self.dim,bias=False))


    def initial_weights(self):
        initializer = nn.init.xavier_uniform_
        self.item_emb = initializer(torch.empty(self.n_item, self.dim))
        self.user_emb = initializer(torch.empty(self.n_user, self.dim))
        self.behavior_emb = initializer(torch.empty(self.n_behavior, self.dim))



    def forward(self, batch,batIds,batIIds,flag):
        
        users = batch['users']
        pos_items = batch['pos_items']
        neg_items = batch['neg_items']
            
        user_bh_embs = []
        item_bh_embs = []

        for i in range(1,self.n_behavior):
            user_embs = [self.user_emb]
            item_embs = [self.item_emb]
            # GCN
            for n in range(self.n_hops):
                bh_mat = self.trainMats[i]
                user_emb = torch._sparse_mm(bh_mat,item_embs[n]*self.behavior_emb[i])
                item_emb = torch._sparse_mm(bh_mat.t(),user_embs[n]*self.behavior_emb[i])
                user_embs.append(user_emb)
                item_embs.append(item_emb)
            
            user_bh_embs.append(self.drop_out(torch.sum(torch.stack(user_embs,0),0)))
            item_bh_embs.append(self.drop_out(torch.sum(torch.stack(item_embs,0),0)))

        # stacking multi-views
        user_bh_embs = torch.stack(user_bh_embs,1)/(self.n_hops)
        item_bh_embs = torch.stack(item_bh_embs,1)/(self.n_hops)

        # RGCN embedding layer
        RGCN_user_embs = self.user_emb
        RGCN_item_embs = self.item_emb
        for _ in range(self.n_hops):
            RGCN_user_embs,RGCN_item_embs = self.Rconv(RGCN_user_embs,RGCN_item_embs,self.trainMats[1:])
    

        # behavior perceptron
        agg_w_u = self.conbine(user_bh_embs.view(-1,self.dim)).view(-1,(self.n_behavior-1))
        agg_w_i = self.conbine(item_bh_embs.view(-1,self.dim)).view(-1,(self.n_behavior-1))
        agg_w_u = F.softmax(agg_w_u,dim=1)
        agg_w_i = F.softmax(agg_w_i,dim=1)
        user_emb = torch.einsum('nb,nbd->nd',[agg_w_u,user_bh_embs]) 
        item_emb = torch.einsum('nb,nbd->nd',[agg_w_i,item_bh_embs])
        user_emb = self.activate(user_emb)
        item_emb = self.activate(item_emb)


        # MLP for ablation
        # agg_w_u = 0
        # user_bh_embs = user_bh_embs.view(-1,self.dim*(self.n_behavior-1))
        # item_bh_embs = item_bh_embs.view(-1,self.dim*(self.n_behavior-1))
        # user_emb = self.activate(self.FC1(user_bh_embs))
        # item_emb = self.activate(self.FC1(item_bh_embs))

        # droupout  
        user_emb = self.layernorm(self.drop_out(user_emb)) 
        item_emb = self.layernorm(self.drop_out(item_emb))
        RGCN_user_embs = self.layernorm(self.drop_out(RGCN_user_embs))
        RGCN_item_embs = self.layernorm(self.drop_out(RGCN_item_embs))
        
        
        if flag =='Train':
            pos_scores =[]
            neg_scores =[]
            

            # multi-task
            for i in range(self.n_behavior-1):
                u_emb = user_emb[users[i]]
                p_emb = item_emb[pos_items[i]]
                n_emb = item_emb[neg_items[i]]

              
                neg_nums = int(neg_items[i].shape[0]/pos_items[i].shape[0])
                pos_scores.append((self.projects[i](u_emb)*p_emb).sum(dim=-1))
                neg_scores.append((self.projects[i](u_emb)*(n_emb.reshape((-1,neg_nums,self.dim)).permute(1,0,2))).sum(dim=-1))
            pos_scores = torch.cat(pos_scores,0)
            neg_scores = torch.cat(neg_scores,1)

            regularizer = (torch.norm(u_emb) ** 2
                    + torch.norm(p_emb) ** 2
                    + torch.norm(n_emb) ** 2) / u_emb.shape[0]
            
            
            loss_c = self._ssl_loss(user_emb,RGCN_user_embs,batIds)*self.ls + self._ssl_loss(item_emb,RGCN_item_embs,batIIds)

            

            return pos_scores, neg_scores, regularizer*0.001+loss_c, agg_w_u
        else:
            # target behavior in Test
            u_emb = user_emb[users]
            p_emb = item_emb[pos_items]
            n_emb = item_emb[neg_items]

            neg_nums = int(neg_items.shape[0]/pos_items.shape[0])
            pos_scores = (self.projects[-1](u_emb)*p_emb).sum(dim=-1)
            neg_scores = (self.projects[-1](u_emb)*(n_emb.reshape((-1,neg_nums,self.dim)).permute(1,0,2))).sum(dim=-1)
            return pos_scores, neg_scores
    

    def cal_loss_L(self,pos_scores,neg_scores):
        neg_nums = neg_scores.shape[0]

        # negative samples treating
        pos_scores= pos_scores.repeat(neg_nums,1).view(-1)
        neg_scores = neg_scores.view(-1)

        loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        return loss
    


    def _ssl_loss(self,current_embeddings,previous_embeddings_all,index):
        current_embeddings = current_embeddings[index]
        previous_embeddings = previous_embeddings_all[index]
        norm_emb1 = F.normalize(current_embeddings)
        norm_emb2 = F.normalize(previous_embeddings)
        norm_all_emb = F.normalize(previous_embeddings_all)
        pos_score = torch.mul(norm_emb1, norm_emb2).sum(dim=1)
        ttl_score = torch.matmul(norm_emb1, norm_all_emb.transpose(0, 1))
        pos_score = torch.exp(pos_score / 0.2)
        ttl_score = torch.exp(ttl_score / 0.2).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()

        return ssl_loss*1e-6
    

    def _get_sparse_adj(self,adj_mat):
        i = torch.LongTensor([adj_mat.row,adj_mat.col]).to(self.device)
        v = torch.FloatTensor(adj_mat.data).to(self.device)
        return torch.sparse.FloatTensor(i, v, adj_mat.shape)


        






       
