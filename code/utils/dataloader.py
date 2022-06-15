import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import numpy as np
import pickle
import pandas as pd
import os.path as osp
import itertools
import os
from icecream import ic


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, rating_file, sep=',', sufix='', transform=None, pre_transform=None):

        self.path = root
        self.dataset = dataset
        self.rating_file = rating_file
        self.sep = sep
        self.sufix=sufix
        self.store_backup = True

        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.stat_info = torch.load(self.processed_paths[1])
        self.data_num = self.stat_info['data_num']
        self.feature_num = self.stat_info['feature_num']

    @property
    def raw_file_names(self):
        return ['{}{}/user_dict.pkl'.format(self.path, self.dataset),
                '{}{}/item_dict.pkl'.format(self.path, self.dataset),
                '{}{}/feature_dict.pkl'.format(self.path, self.dataset),
                '{}{}/{}'.format(self.path, self.dataset, self.rating_file)]

    @property
    def processed_file_names(self):
        return ['{}/{}.dataset'.format(self.dataset+self.sufix, self.dataset),
                '{}/{}.statinfo'.format(self.dataset+self.sufix, self.dataset)]


    def download(self):
        # Download to `self.raw_dir`.
        pass


    def data_2_graphs(self, ratings_df, dataset='train'):
        graphs = []
        graphs_pos = []
        graphs_neg = []
        processed_graphs = 0
        num_graphs = ratings_df.shape[0]
        one_per = int(num_graphs/1000)
        percent = 0.0
        for i in range(len(ratings_df)):
            if processed_graphs % one_per == 0:
                print(f"Processing [{dataset}]: {percent/10.0}%, {processed_graphs}/{num_graphs}", end="\r")
                percent += 1
            processed_graphs += 1 
            line = ratings_df.iloc[i]
            user_index = self.user_key_type(line[0])
            item_index = self.item_key_type(line[1])
            rating = int(line[2])

            if item_index not in self.item_dict or user_index not in self.user_dict:
                error_num += 1
                continue

            user_id = self.user_dict[user_index]['name']
            item_id = self.item_dict[item_index]['title']

            user_attr_list = self.user_dict[user_index]['attribute']
            item_attr_list = self.item_dict[item_index]['attribute']

            #user_list = [user_id] + user_attr_list
            #item_list = [item_id] + item_attr_list
            feature_list = [user_id, item_id] + user_attr_list + item_attr_list

            graph = self.construct_graph(feature_list, rating)
            if dataset == 'train':
                if rating > 0:
                    graphs_pos.append(graph)
                else:
                    graphs_neg.append(graph)
            else:
                graphs.append(graph)

        print()

        if dataset == 'train':
            return graphs_pos, graphs_neg
        else:
            return graphs



    def read_data(self):
        self.user_dict = pickle.load(open(self.userfile, 'rb'))
        self.item_dict = pickle.load(open(self.itemfile, 'rb'))
        self.user_key_type = type(list(self.user_dict.keys())[0])
        self.item_key_type = type(list(self.item_dict.keys())[0])
        feature_dict = pickle.load(open(self.featurefile, 'rb'))

        data = []
        error_num = 0

        ratings_df = pd.read_csv(self.ratingfile, sep=self.sep, header=None)
        #train_df, test_df = train_test_split(ratings_df, test_size=0.4, random_state=2019, stratify=ratings_df[[0, 2]])
        #test_df, valid_df = train_test_split(test_df,  test_size=0.5, random_state=2019, stratify=test_df[[0, 2]])
        train_df, test_df = train_test_split(ratings_df, test_size=0.15, random_state=2019, stratify=ratings_df[[0,2]])
        train_df, valid_df = train_test_split(train_df,  test_size=15/85, random_state=2019, stratify=train_df[[0,2]])

        # store a backup of train/valid/test dataframe
        if self.store_backup:
            backup_path = f"{self.path}{self.dataset}/split_data_backup/"
            if not os.path.exists(backup_path):
                os.mkdir(backup_path)

            train_df.to_csv(f'{backup_path}train_data.csv', index=False)
            valid_df.to_csv(f'{backup_path}valid_data.csv', index=False)
            test_df.to_csv(f'{backup_path}test_data.csv', index=False)


        train_graphs_p, train_graphs_n = self.data_2_graphs(train_df, dataset='train')
        valid_graphs = self.data_2_graphs(valid_df, dataset='valid')
        test_graphs = self.data_2_graphs(test_df, dataset='test')

        graphs = train_graphs_p + train_graphs_n + valid_graphs + test_graphs 

        stat_info = {}
        stat_info['data_num'] = len(graphs)
        stat_info['feature_num'] = len(feature_dict)
        len_p = len(train_graphs_p)
        len_n = len(train_graphs_n)
        len_valid = len(valid_graphs)
        stat_info['train_test_split_index'] = [len_p, len_p+len_n, len_p+len_n+len_valid]

        print('error number of data:', error_num)
        return graphs, stat_info


    def construct_graph(self, node_list, rating):
        x = torch.LongTensor(node_list).unsqueeze(1)
        rating = torch.FloatTensor([rating])
        return Data(x=x, y=rating)

    def process(self):
        self.userfile  = self.raw_file_names[0]
        self.itemfile  = self.raw_file_names[1]
        self.featurefile = self.raw_file_names[2]
        self.ratingfile  = self.raw_file_names[3]
        graphs, stat_info = self.read_data()

        #check whether foler path exist
        if not os.path.exists(f"{self.path}processed/{self.dataset+self.sufix}"):
            os.mkdir(f"{self.path}processed/{self.dataset+self.sufix}")


        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

        torch.save(stat_info, self.processed_paths[1])

    def node_M(self):
        return self.feature_num

    def data_N(self):
        return self.data_num


