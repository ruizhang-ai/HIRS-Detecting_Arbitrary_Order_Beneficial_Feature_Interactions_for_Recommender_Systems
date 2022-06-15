import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import argparse
from utils.dataloader import Dataset
from torch_geometric.data import DataLoader

from torch_geometric.nn import global_mean_pool
from models import HyperInfomax as MODEL
from utils import process
from icecream import ic
import torch.autograd.profiler as profiler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import statistics as st

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m', help='which dataset to use')
parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--l0_para', nargs='?', default='[0.66, -0.1, 1.1]',
                        help="l0 parameters, which are beta (temprature), \
                            zeta (interval_min) and gama (interval_max).")
parser.add_argument('--lbd', nargs='?', default="[1,0.1, 0.02, 0.00001]",
    help='coef for all loss, order: pred, infomax, infomin, l0, l2')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
parser.add_argument('--hid_units', type=int, default=64, help='neural hidden layer')
parser.add_argument('--n_neg_max', type=int, default=1, help='the number of negative samples for infomax')
parser.add_argument('--n_neg_min', type=int, default=1, help='the number of negative samples for informin')
parser.add_argument('--edge_num', type=int, default=40, help='The number of predicted edges of each data')
parser.add_argument('--random_seed', type=int, default=2019, help='random seeds')
parser.add_argument('--patience', type=int, default=5, help='patience to stop training')


args = parser.parse_args()

if args.dataset in ['ml-1m']:
    args.num_features = 14
elif args.dataset == 'book-crossing':
    args.num_features = 39 
elif args.dataset == 'ml-25m':
    args.num_features = 15 
else:
    print(f"please specify number of features for dataset {args.dataset}")
    exit()

start_time = datetime.now()
writer = SummaryWriter(f"../results/runs/train_together_{datetime.now()}")

"""
=========================================================
Data Loading
=========================================================
"""
dataset = Dataset('../data/', args.dataset, 'implicit_ratings.csv')
n_features = dataset.node_M() 
data_num = dataset.data_N()
train_p = 0.7
test_p = (1 - train_p)/2 

pos_num, neg_num, valid_num = dataset.stat_info['train_test_split_index']
train_dataset_pos = dataset[:pos_num]
train_dataset_neg = dataset[pos_num:neg_num]
val_dataset = dataset[neg_num:valid_num]
test_dataset = dataset[valid_num:]

# calculate batch size for negative data samples
n_round = int(len(train_dataset_pos)/args.batch_size)
neg_batch_size = int(len(train_dataset_neg)/n_round)

nw=8 
train_loader_pos = DataLoader(train_dataset_pos, shuffle=True, batch_size=args.batch_size, num_workers=nw)
train_loader_neg = DataLoader(train_dataset_neg, shuffle=True, batch_size=neg_batch_size, num_workers=nw)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, num_workers=nw)
#val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, num_workers=nw)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Runing on device {device}")


"""
=========================================================
Model Loading and coef setting
=========================================================
"""
#model = HyperInfomax(args, n_features, device, writer).to(device)
#model = InfomaxNFM(args, n_features, device, writer).to(device)
model = MODEL(args, n_features, device, writer).to(device)
lbd = eval(args.lbd)


b_xent = nn.BCELoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best_ndcg = 0 
best_t = 0



def evaluation(model, data_loader, device, is_test=False, epoch=-1):
    model.eval()
    predictions = []
    labels = []
    user_ids = []
    #user_ids = []
    record = True
    for data in data_loader:
        data = data.to(device)
        node_index = torch.squeeze(data.x)
        edge_index = data.edge_index
        batch = data.batch
        #user_id = node_index.view((batch.max()+1), -1)[:,0].detach().cpu().numpy()
        _, user_id_index = np.unique(batch.detach().cpu().numpy(), return_index=True)
        user_id = data.x.detach().cpu().numpy()[user_id_index]
        y = data.y
        if epoch != -1 and record:
            pred, _ = model.run_pred((node_index, edge_index, batch), False, True, epoch)
            record = False
        else:
            pred, _ = model.run_pred((node_index, edge_index, batch), False)
        
        pred = pred.squeeze().detach().cpu().numpy().astype('float64')
        y = y.detach().cpu().numpy()

        predictions.append(pred)
        labels.append(y)
        user_ids.append(user_id)

    return process.eval_metrics(predictions, labels, user_ids, is_test)
"""
optimiser = torch.optim.Adagrad(
        model.parameters(),
        args.lr,
        lr_decay=lbd[-1],
        #weight_decay=1e-5
    )
"""
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=lbd[-1])


writer_scalar_names = ['Total Loss', 'Base Loss', 'Infomax Loss', 'Infomin Loss', 'L0 Loss', 'edges']
for epoch in range(args.n_epochs):
    """
    =========================================================
    Training
    =========================================================
    """
    index = 0
    #with profiler.profile(use_cuda=True, record_shapes=True) as prof:
    record = True
    t_loss_list = []
    base_loss_list = []
    t_distance_c = []
    t_distance_h = []
    l0_loss_list= []
    edge_list = []
    infomax_loss_list = []
    infomin_loss_list = []
    writer_scalar_values = [0 for i in range(len(writer_scalar_names))]
    for data_p, data_n in zip(train_loader_pos, train_loader_neg):
        data_p = data_p.to(device)
        data_n = data_n.to(device)
        #if index == 3:
        #    break
        index += 1 
        model.train()
        optimiser.zero_grad()

        node_index_p = torch.squeeze(data_p.x)
        node_index_n = torch.squeeze(data_n.x)
        edge_index_p = data_p.edge_index
        edge_index_n = data_n.edge_index
        batch_p = data_p.batch
        batch_n = data_n.batch
        y_p = data_p.y    # all 1 
        y_n = data_n.y    # all 0
            #with profiler.record_function("model_inference"):
        #with profiler.profile(use_cuda=True, record_shapes=True) as prof:
        #whole_out, infomax_out, pred_out, lbl, l0_penaty = model(node_index, batch, True)
        pos_data = (node_index_p, edge_index_p, batch_p)
        neg_data = (node_index_n, edge_index_n, batch_n)
        pred_p, pred_n, loss_infomax, loss_infomin, (distance_c, distance_h), (l0_loss, n_edges), (edge_stats_p, edge_stats_n)\
                = model(pos_data, neg_data, True, record=(index==1), epoch=epoch)
        
        #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

        # write edge statistics to tensorboard
        """
        Loss Part 
        """

        loss_pred_p = b_xent(pred_p, y_p) 
        loss_pred_n = b_xent(pred_n, y_n) 
        #loss = lbd[0] * loss_whole + lbd[1] * loss_infomax + lbd[2] * loss_pred + lbd[3] * l0_penaty
        #loss = lbd[1] * loss_infomax + lbd[2] * loss_pred + lbd[3] * l0_penaty
        baseloss = (loss_pred_p + loss_pred_n)
        loss = baseloss + lbd[0]*loss_infomax + lbd[1] * loss_infomin + lbd[2] * l0_loss 

        t_loss_list.append(loss)
        base_loss_list.append(baseloss)
        #t_distance_c.append(distance_c)
        #t_distance_h.append(distance_h)
        l0_loss_list.append(l0_loss)
        edge_list.append(n_edges)
        infomax_loss_list.append(loss_infomax)
        infomin_loss_list.append(loss_infomin)

        loss.backward()
        optimiser.step()

    mean_loss = sum(t_loss_list)/len(t_loss_list)
    mean_baseloss = sum(base_loss_list)/len(base_loss_list)
    #mean_distance_c = sum(t_distance_c)/len(t_distance_c)
    #mean_distance_h = sum(t_distance_h)/len(t_distance_h)
    mean_l0_loss = sum(l0_loss_list)/len(l0_loss_list)
    mean_edges = sum(edge_list)/len(edge_list)
    mean_infomax_loss = sum(infomax_loss_list)/len(infomax_loss_list)
    mean_infomin_loss = sum(infomin_loss_list)/len(infomin_loss_list)

    """
    =========================================================
    Evaluation 
    =========================================================
    """
    ndcgs = evaluation(model, val_loader, is_test=False,  device=device)
    print(f"Epoch {epoch}: loss: {mean_loss:.6f}, baseloss:{mean_baseloss:.6f}, infomax: {mean_infomax_loss:.6f}, infomin: {mean_infomin_loss:.6f}, ", end=' ')
    print(f"val NDCG: {ndcgs[0]:.6f}, {ndcgs[1]:.6f}, {ndcgs[2]:.6f}", end=' ') 
    print(f"l0_loss: {mean_l0_loss:.6f}, edges: {mean_edges:.1f}")

    if ndcgs[1] > best_ndcg:
        best_ndcg = max(ndcgs[1], best_ndcg) 
        best_t = epoch
        cnt_wait = 0
        #end_time = datetime.now()
        torch.save(model.state_dict(), f'../results/best_{args.dataset}_{start_time}.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == args.patience:
        print('Early stopping!')
        break


print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load(f'../results/best_{args.dataset}_{start_time}.pkl'))
test_ndcgs, test_recalls = evaluation(model, test_loader, is_test=True, device=device)
print(f"The test results:")
print(f"ndcg5: {test_ndcgs[0]}\t ndcg10: {test_ndcgs[1]}, ndcg20: {test_ndcgs[2]}")
print(f"recall@5: {test_recalls[0]}\t recall@10: {test_recalls[1]}, recall@20: {test_recalls[2]}")


