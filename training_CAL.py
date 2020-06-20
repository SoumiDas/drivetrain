# standard imports 
import numpy as np
import torch
#import matplotlib.pyplot as plt
from torch import optim
#from ipdb import set_trace


# own modules
from net_actual import get_model
from dataloader_actual import CAL_Dataset
from dataloader_actual import get_data, get_mini_data
from train_actual import fit, custom_loss, validate
from metrics_actual_changed import calc_metrics


# paths
data_path = 'dataset/'
data_path

if not os.path.exists('models'):
	os.mkdir('models')

if not os.path.exists('total_models'):
	os.mkdir('total_models')

params = {'name': 'CAL_whole_multigpu', 'type_': 'LSTM', 'lr': 1e-4, 'n_h': 100, 'p':0.44, 'seq_len':10}
model, opt = get_model(params)
print('Model got')

device = torch.device('cuda')

if torch.cuda.device_count() > 1:
  modelmg = torch.nn.DataParallel(model,device_ids=[0, 1, 2, 3]).to(device)

#model = model.to(device)

train_dl, valid_dl = get_data(data_path, params['seq_len'], batch_size=10)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = model.to(device)


#train_dl, valid_dl = get_data(data_path, model.params.seq_len, batch_size=10)

model, val_hist = fit(50, modelmg, params, custom_loss, opt, train_dl, valid_dl)
