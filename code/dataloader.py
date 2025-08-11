import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
import torch.utils
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class Loader(torch.utils.data.Dataset):

    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.extraUser, self.extraItem = [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.extraDataSize = 0

        genre_file_path = path + '/genres.txt'
        self.game_to_genres, self.all_genres = self.load_game_genres(genre_file_path)

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)

        self.m_item += 1
        self.n_user += 1
        
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.Graph = None

        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                    shape=(self.n_user, self.m_item))
        
        if self.extraDataSize > 0:
            self.ExtraUserItemNet = csr_matrix((np.ones(len(self.extraUser)), 
                                            (self.extraUser, self.extraItem)),
                                            shape=(self.n_user, self.m_item))

            self.AggregationUserItemNet = self.UserItemNet.copy()

            self.AggregationUserItemNet = self.AggregationUserItemNet.maximum(self.ExtraUserItemNet)
        else:
            self.ExtraUserItemNet = None
            self.AggregationUserItemNet = self.UserItemNet.copy()
        
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        self.pop_threshold = np.percentile(self.items_D, 80)
        print(f"Popularity threshold (80th percentile): {self.pop_threshold}")
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")
        

        self.ii_path = path + '/checkpoint.txt'

        self.prepare_training_edges()
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
 
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_with_extra.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()

                R = self.AggregationUserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat_with_extra.npz', norm_adj)
            
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world.device)
            print("don't split the matrix")
        return self.Graph
    

    
    def getOriginalUserPosItems(self, users):

        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    

    def load_game_genres(self, genre_file_path):

        game_to_genres = {}
        all_genres = set()
        
        with open(genre_file_path, 'r') as f:
            for line in f.readlines():
                if len(line.strip()) > 0:
                    game_id, genre = line.strip().split(',')
                    game_id = int(game_id)
                    
                    if game_id not in game_to_genres:
                        game_to_genres[game_id] = set()
                    
                    game_to_genres[game_id].add(genre)
                    all_genres.add(genre)
        
        print(f"Successfully loaded category information for {len(game_to_genres)} games")
        print(f"Total of {len(all_genres)} different game categories")
        
        return game_to_genres, all_genres
    

        



    def load_conversion_rate_data(self, conversion_file_path):

            print(f"Loading item conversion rate data [{conversion_file_path}]")
            item_item_conversion = {}
            
            try:
                with open(conversion_file_path, 'r') as f:
                    
                    header = f.readline()
                    for line in f.readlines():
                        if len(line.strip()) > 0:
                            parts = line.strip().split(',')
                            if len(parts) == 3:  
                                source = int(parts[0])
                                target = int(parts[1])
                                rate = float(parts[2])
                                
                                if target not in item_item_conversion:
                                    item_item_conversion[target] = {}
                                
                                item_item_conversion[target][source] = rate
                    
                    print(f"Successfully loaded conversion rate data for {len(item_item_conversion)} items")
                    return item_item_conversion
                
            except FileNotFoundError:
                print(f"Conversion rate file {conversion_file_path} does not exist")
                return {}


    def getIIGraph(self):
        print("Loading item conversion rate adjacency matrix")
        if not hasattr(self, 'ConversionGraph') or self.ConversionGraph is None:
            try:
                pre_conv_mat = sp.load_npz(self.path + '/s_pre_conversion_graph.npz')
                print("Successfully loaded item conversion rate matrix...")
                conv_graph = pre_conv_mat
            except:
                print("Generating item conversion rate adjacency matrix")
                s = time()
                
                conv_graph = sp.dok_matrix((self.m_items, self.m_items), dtype=np.float32)

                ii_conversion = self.load_conversion_rate_data(self.ii_path)
                for target, sources in ii_conversion.items():
                    for source, rate in sources.items():
                        if source < self.m_items and target < self.m_items:
                            conv_graph[target, source] = rate
                
                rowsum = np.array(conv_graph.sum(axis=1))
                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_conv_graph = d_mat.dot(conv_graph)
                norm_conv_graph = norm_conv_graph.tocsr()
                
                end = time()
                print(f"Time cost: {end-s}s, saving item conversion rate matrix...")
                sp.save_npz(self.path + '/s_pre_conversion_graph.npz', norm_conv_graph)
                
                conv_graph = norm_conv_graph
            
            self.ConversionGraph = self._convert_sp_mat_to_sp_tensor(conv_graph)
            self.ConversionGraph = self.ConversionGraph.coalesce().to(world.device)
            
            edge_count = self.ConversionGraph._nnz()
            print(f"Item conversion graph shape: {self.ConversionGraph.shape}, non-zero edges: {edge_count}")
        
        return self.ConversionGraph


