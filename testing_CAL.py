

# standard imports 
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import optim
#from ipdb import set_trace


# own modules
#from dataloader_time_new_seq import CAL_Dataset
#from dataloader_separate import CAL_Dataset
#from net_Indranil_center import get_model
from net_actual import get_model
from dataloader_actual import CAL_Dataset
from dataloader_actual import get_data, get_mini_data
from train_actual import fit, custom_loss, validate
from metrics_actual_changed import calc_metrics
#from net_my_CAL_test import get_model
#from net_concat import get_model
#from net_concat_RC import get_model
#from dataloader_crop_6825 import CAL_Dataset
#from dataloader_crop_6825 import get_data, get_mini_data
#from dataloader_time_new_seq import get_data, get_mini_data
#from dataloader_separate import get_data, get_mini_data
#from train_6825_new import fit, custom_loss, validate
#from train_time_new_RC import fit, custom_loss, validate
#from train_separate import fit, custom_loss, validate
#from metrics_Indranil_center import calc_metrics
#from metrics_6825_SMPE import calc_metrics
#from metrics_separate import calc_metrics

# paths
data_path = '../../new_data_Episodes/Episodes/'
data_path


# In[5]:



#print(t)


# In[ ]:





# In[ ]:


# manualSeed = 42

# np.random.seed(manualSeed)
# torch.manual_seed(manualSeed)

# # if you are using GPU
# torch.cuda.manual_seed(manualSeed)
# torch.cuda.manual_seed_all(manualSeed)

# torch.backends.cudnn.enabled = False 
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


# #### Training

# Initialize the model. Possible Values for the task block type: MLP, LSTM, GRU, TempConv

# In[ ]:


#params = {'name': '1868_psi_1', 'type_': 'LSTM', 'lr': 1e-4, 'n_h': 100, 'p':0.44, 'seq_len':10}
#params = {'name': 'many_episodes_whole_1', 'type_': 'GRU', 'lr': 1e-4, 'n_h': 100, 'p':0.3, 'seq_len':10}
#params = {'name': 'many_episodes_trial', 'type_': 'GRU', 'lr': 1e-4, 'n_h': 100, 'p':0.3, 'seq_len':10}
#params = {'name': 'many_episodes_more_trial2', 'type_': 'GRU', 'lr': 1e-4, 'n_h': 100, 'p':0.3, 'seq_len':10}
params = {'name': 'new_data_CAL_whole_vgg19_sameparams_new', 'type_': 'GRU', 'lr': 1e-4, 'n_h': 100, 'p':0.3, 'seq_len':10}
model, opt = get_model(params)
print('Model got')
#params


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[ ]:


#a = np.load('/content/drive/My Drive/CAL-master/training/To_be_done_early/CameraRGB/is_train.npy')
#b = np.load('/content/drive/My Drive/CAL-master/training/To_be_done_early/CameraRGB/is_test.npy')
#c = np.load('/content/drive/My Drive/CAL-master/training/To_be_done_early/CameraRGB/is_val.npy')
#print(len(a))
#print(len(b))
#print(len(c))
#a = np.delete(a, 1, 0)
#b = np.delete(b, 1, 0)
#c = np.delete(c, 1, 0)
#print(len(a))
#print(len(b))
#print(len(c))
#np.save('/content/drive/My Drive/CAL-master/training/To_be_done_early//CameraRGB/is_train.npy', a)
#np.save('/content/drive/My Drive/CAL-master/training/To_be_done_early//CameraRGB/is_test.npy', b)
#np.save('/content/drive/My Drive/CAL-master/training/To_be_done_early//CameraRGB/is_val.npy', c)


# In[ ]:


#a = np.load('/content/drive/My Drive/CAL-master/training/To_be_done_early/CameraRGB/is_train.npy')
#b = np.load('/content/drive/My Drive/CAL-master/training/To_be_done_early/CameraRGB/is_test.npy')
#c = np.load('/content/drive/My Drive/CAL-master/training/To_be_done_early/CameraRGB/is_val.npy')
#print(len(a))
#print(len(b))
#print(len(c))
#np.insert(a, 1, 'True')

#a = np.insert(a, 1, 'True')
#print(len(a))
#np.save('/content/drive/My Drive/CAL-master/training/To_be_done_early//CameraRGB/is_train.npy', a)


# In[ ]:


#print(len(a))
#print(len(b))
#print(len(c))
#pwd


# get the data loader. get mini data gets only a subset of the training data, on which we can try if the model is able to overfit

# In[ ]:


train_dl, valid_dl = get_data(data_path, model.params.seq_len, batch_size=10)
print(len(train_dl))
print(len(valid_dl))
#print(valid_dl)
# train_dl, valid_dl = get_mini_data(data_path, model.params.seq_len, batch_size=16, l=4000)


# In[7]:


#train_dl = get_data(data_path, model.params.seq_len, batch_size=16)


# uncomment the next cell if the feature extractor should also be trained

# In[ ]:


# for name,param in model.named_parameters():
#     param.requires_grad = True
# opt = optim.Adam(model.parameters())


# Train the model. We automatically save the model with the lowest val_loss. If you want to continue the training and keep the loss history, just pass it as an additional argument as shown below.

# In[9]:


#get_ipython().system('export CUDA_LAUNCH_BLOCKING = 1;')


# In[ ]:


'''val_hist = [0.06873621344566345, 0.06517140234217927, 0.05959741990355886, 0.05723178623353733, 0.055425322494086084, 
           0.05185591017498689, 0.049858507075730474, 0.048336263263926774, 0.0486204710076836, 0.047835044650470504, 
           0.04648559908656514, 0.0458745564608013, 0.04595366278115441, 0.04595366278115441, 0.04603604832116295, 
           0.04489362511564703, 0.04527037915061503, 0.044825669684830836, 0.04476596651708379]'''


# In[ ]:


#from multiprocessing import Process, Queue
#from threading import Thread

#class someClass(object):
 #   def __init__(self):
 #       pass
 #   def f(self, x):
 #       return x*x

 #   def go(self):
 #       p = Pool(4)
 #       sc = p.map(self, range(4))
 #       print(sc)

 #   def __call__(self, x):   
 #       return self.f(x)

#sc = someClass()
#sc.go()
#!pip install torch
#from torch import optim

#model, val_hist = fit(30, model, custom_loss, opt, train_dl, valid_dl)
model, _ = get_model(params)
model.load_state_dict(torch.load(f"./total_models/{model.params.name}.pth"))
model.eval().to(device);
#model = model.to(device);
print(model.params.name)

model.eval();
_,_, preds, labels= validate(model, valid_dl, custom_loss)

discrete, mae, mape, smpe, maef1 = calc_metrics(preds, labels)

int2key = {0: 'red_light', 1:'hazard_stop', 2:'speed_sign', 3:'relative_angle', 4: 'center_distance', 5: 'veh_distance'}
		
for kv in range(6):

	k = int2key[kv]
	class_labels = ['red_light', 'hazard_stop', 'speed_sign']
	all_preds = np.argmax(preds[k], axis=1) if k in class_labels else preds[k]
	all_labels = labels[k][:, 1] if k in class_labels else labels[k]	
	with open('preds_gt_analyse_best_'+str(k)+'.txt','a') as fppgv:
		for i in range(len(all_preds.tolist())):
			fppgv.write(str(all_preds[i])+','+str(all_labels[i])+'\n')
			


print(discrete)
print('\n')
print(mae)
print('\n')
print(mape)
print('\n')
print(smpe)
print('\n')
print(maef1)

'''with open('preds_gt_relative_newdata_CAL_whole.txt','w') as fp:
   for i in range(len(all_preds['relative_angle'].tolist())):
      fp.write(str(all_preds['relative_angle'][i][0])+','+str(all_labels['relative_angle'][i])+'\n')
      
with open('preds_gt_centerline_newdata_CAL_whole.txt','w') as fp:
   for i in range(len(all_preds['center_distance'].tolist())):
      fp.write(str(all_preds['center_distance'][i][0])+','+str(all_labels['center_distance'][i])+'\n')

with open('preds_gt_redlight_newdata_CAL_whole.txt','w') as fp:
   for i in range(len(all_preds['red_light'].tolist())):
      fp.write(str(all_preds['red_light'][i][0])+','+str(all_labels['red_light'][i])+'\n')
      
with open('preds_gt_vehicle_newdata_CAL_whole.txt','w') as fp:
   for i in range(len(all_preds['veh_distance'].tolist())):
      fp.write(str(all_preds['veh_distance'][i][0])+','+str(all_labels['veh_distance'][i])+'\n')
      
with open('preds_gt_speed_newdata_CAL_whole.txt','w') as fp:
   for i in range(len(all_preds['speed_sign'].tolist())):
      fp.write(str(all_preds['speed_sign'][i][0])+','+str(all_labels['speed_sign'][i])+'\n')
      
with open('preds_gt_hazard_newdata_CAL_whole.txt','w') as fp:
   for i in range(len(all_preds['hazard_stop'].tolist())):
      fp.write(str(all_preds['hazard_stop'][i][0])+','+str(all_labels['hazard_stop'][i])+'\n')'''
      
  

'''discrete, mae, mape, smpe, maef1 = calc_metrics(all_preds, all_labels)

print(discrete)
print('\n')
print(mae)
print('\n')
print(mape)
print('\n')
print(smpe)
print('\n')
print(maef1)'''

# for convience, we can pass an integer instead of the full string
'''int2key = {0: 'red_light', 1:'hazard_stop', 2:'speed_sign', 
           3:'relative_angle', 4: 'center_distance', 5: 'veh_distance'}


# In[ ]:


def plot_preds(k, all_preds, all_labels, start=500, delta=1188):
    if isinstance(k, int): k = int2key[k]
    
    # get preds and labels
    class_labels = ['red_light', 'hazard_stop', 'speed_sign']
    pred = np.argmax(all_preds[k], axis=1) if k in class_labels else all_preds[k]
    label = all_labels[k][:, 1] if k in class_labels else all_labels[k]
    
    print(pred[0])
    plt.plot(pred[start:start+delta], 'r-', label='Prediction', linewidth=2.0)
    plt.plot(label[start:start+delta], 'g', label='Ground Truth', linewidth=2.0)
    
    plt.legend()
    plt.grid()
    plt.savefig('relative_subset_500_1188.png')
    plt.show()


# In[25]:


plot_preds(3, all_preds, all_labels, start=0, delta=1188)'''

# In[ ]:




