# standard imports 
import numpy as np
import torch
#import matplotlib.pyplot as plt
from torch import optim
#from ipdb import set_trace
import os

# own modules
import ops
from net_actual import get_model
from dataloader_actual import CAL_Dataset
from dataloader_actual import get_data, get_mini_data
from train_actual import fit, custom_loss, validate
from metrics_actual_changed import calc_metrics
import time

# paths
data_path = './transformed_CAL/'
#data_path
print("Path")
print(os.getcwd())
if not os.path.exists('./models/'):
	print("Models")
	os.makedirs('models')

if not os.path.exists('./total_models/'):
	os.makedirs('total_models')

params = {'name': 'CAL_whole_multigpu_trans', 'type_': 'LSTM', 'lr': 1e-4, 'n_h': 100, 'p':0.44, 'seq_len':10}
model, opt = get_model(params)
print('Model got')

if os.path.exists("./total_models/"+params['name']+".pth"):
	print("Loading saved weights")
	model.load_state_dict(torch.load("./total_models/"+params['name']+".pth",map_location='cpu'))

#model.eval().to(device);

device = torch.device('cuda')

print("Cuda device count is: ")
print(torch.cuda.device_count())

#print(device)

if torch.cuda.device_count() > 1:
  model = torch.nn.DataParallel(model,device_ids=[0, 1, 2, 3]).to(device)

#print("Model device")
#print(model.device)
model = model.to(device)

print("Data loading")
start2 = time.time()
train_dl, valid_dl = get_data(data_path, params['seq_len'], batch_size=9)
print(time.time() - start2)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = model.to(device)


#train_dl, valid_dl = get_data(data_path, model.params.seq_len, batch_size=10)

model, val_hist = fit(50, model, params, custom_loss, opt, train_dl, valid_dl)
