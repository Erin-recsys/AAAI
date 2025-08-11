import world
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import time
import os
class HyperConv(nn.Module):

    def __init__(self, config):
        super(HyperConv, self).__init__()
        self.layers = config['ii_layers'] 
        #self.weight = config['ii_weight']  
        
    def forward(self, ii_graph, item_embeddings):

        initial_embeddings = item_embeddings
        all_embeddings = [initial_embeddings]
        
        current_embeddings = initial_embeddings
        for i in range(self.layers):
            aggregated = torch.sparse.mm(ii_graph, current_embeddings)
            all_embeddings.append(aggregated)
            current_embeddings = aggregated
    
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        return final_embeddings
class LightGCN(nn.Module):
    def __init__(self, 
                 config:dict, 
                 dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')

        item_degrees = np.array(self.dataset.items_D)
        self.pop_threshold = np.percentile(item_degrees, 80)
        print(f"Popularity threshold: {self.pop_threshold}")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
        print(f"Graph的shape{self.Graph.shape}")

        self.special_interactions = {}
        self.special_weights = {}  

        try:
            file_path = f"./data/{self.dataset.path.split('/')[-1]}/two_way_counterfactual_popularity_{self.config['top_k']}_{self.config['top_m']}.txt"
            print(file_path)

            with open(file_path, 'r') as f:
                for line in f.readlines():
                    if len(line.strip()) > 0:
                        items = line.strip().split()
                        if len(items) > 1:
                            user_id = int(items[0])
                            self.special_interactions[user_id] = []
                            self.special_weights[user_id] = {}  

                            for item_weight in items[1:]:

                                item_id, weight = item_weight.split(":")
                                item_id = int(item_id)
                                weight = float(weight)
                                self.special_interactions[user_id].append(item_id)
                                self.special_weights[user_id][item_id] = weight
            
            print(f"Successfully loaded special interaction data for {len(self.special_interactions)} users")
            
            user_in_range = [u for u in self.special_interactions.keys() if u < self.num_users]
            print(f"Valid user ID count: {len(user_in_range)}/{len(self.special_interactions)}")

        except Exception as e:
            print(f"Failed to load special interaction data: {e}")

        self.HyperConv = HyperConv(self.config)
        self.ii_graph = self.dataset.getIIGraph()



    def computer(self):

        users_emb = self.embedding_user.weight 
        items_emb = self.embedding_item.weight 
        items_emb = self.HyperConv(self.ii_graph, items_emb)
        all_emb = torch.cat([users_emb, items_emb]) 

        embs = [all_emb]
        for layer in range(self.config['lightGCN_n_layers']):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items


    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        #all_items = self.HyperConv(self.ii_graph, all_items)
        if not self.config['counterfactual_popularity']:
            users_emb = all_users[users.long()]
            items_emb = all_items
            rating = self.f(torch.matmul(users_emb, items_emb.t()))
            return rating
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        if not self.training and hasattr(self, 'special_interactions') and self.special_interactions:

            special_mask = torch.ones_like(rating)
            
            users_cpu = users.cpu().numpy()
            
            batch_indices = []
            item_indices = []
            
            for batch_idx, user_id in enumerate(users_cpu):
                if user_id in self.special_interactions:
                    
                    special_items = self.special_interactions[user_id]
                    
                    valid_items = [item for item in special_items if item < special_mask.shape[1]]
                    
                    
                    if valid_items:
                        batch_indices.extend([batch_idx] * len(valid_items))
                        item_indices.extend(valid_items)

            if batch_indices:
                
                batch_indices = torch.tensor(batch_indices, device=special_mask.device)
                item_indices = torch.tensor(item_indices, device=special_mask.device)
                special_mask[batch_indices, item_indices] = self.config['counterfactual_popularity_weight']

            adjusted_rating = rating * special_mask
         
        rating = adjusted_rating#self.f(adjusted_rating)
        
        return rating
  
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]

        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def loss(self, users, pos, neg):

        all_users, all_items = self.computer()

        users_emb = all_users[users.long()]
        pos_emb = all_items[pos.long()]
        neg_emb = all_items[neg.long()]
        
        userEmb0 = self.embedding_user(users.long())
        posEmb0 = self.embedding_item(pos.long())
        negEmb0 = self.embedding_item(neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                        posEmb0.norm(2).pow(2) + 
                        negEmb0.norm(2).pow(2))/float(len(users))
        
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return bpr_loss, reg_loss
       





