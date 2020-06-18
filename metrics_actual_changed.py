import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

root_dir = 'new_data_Episodes/Episodes/'
data_path = 'new_data_Episodes/Episodes/'

def get_intersection_union_per_class(confusion_matrix):
    number_of_labels = confusion_matrix.shape[0]
    matrix_diagonal = [confusion_matrix[i][i] for i in range(number_of_labels)]
    errors_summed_by_row = [0] * number_of_labels
    
    for row in range(number_of_labels):
        for column in range(number_of_labels):
            if row != column:
                errors_summed_by_row[row] += confusion_matrix[row][column]
    errors_summed_by_column = [0] * number_of_labels
    
    for column in range(number_of_labels):
        for row in range(number_of_labels):
            if row != column:
                errors_summed_by_column[column] += confusion_matrix[row][column]
        
    divisor = [0] * number_of_labels
    for i in range(number_of_labels):
        divisor[i] = matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
        if matrix_diagonal[i] == 0:
            divisor[i] = 1
               
    return [float(matrix_diagonal[i]) / divisor[i] for i in range(number_of_labels)]
    
def calculate_scores(cl, pred_cl):
    cm = confusion_matrix(cl, pred_cl)  #  (2,2)  for 'red_light' and 'hazard_stop' ; (4,4) for 'speed_sign'
    precision, recall = [], []
    for i in range(1,len(cm)):
        p1 = 100*(cm[i,i]/(np.sum(cm[:,i])))
        r1 = 100*(cm[i,i]/(np.sum(cm[i,:])))
        precision.append('{:.2f}'.format(p1))
        recall.append('{:.2f}'.format(r1))     
    val_acc = 100*float(cm.trace())/np.sum(cm)  #val_acc => sum of diagonal elements / sum of all elements
    IoUs = get_intersection_union_per_class(cm)
    IoU_mean = 100*np.mean(IoUs)                #IoU_mean => percentage of True positive
        
    return val_acc, IoU_mean
    
def labels2classes(predictions):
    classes = np.argmax(predictions, axis=1)    #returns the index of maximum value in a row
    return classes.reshape((-1,1))           #reshapes a matrix (n,m) into (n*m,1)
    
def calc_metrics(preds, labels):  
    scores_MAE = {}
    scores_MAPE = {}
    scores_SMPE = {}
    scores_discrete = {}
    scores_MAE_F1={}
    #print("it is in the code")
    ### Classification
    classification_labels = ['red_light', 'hazard_stop', 'speed_sign']
    for k in classification_labels:
        cl, pred_cl = labels2classes(labels[k]), labels2classes(preds[k])
        scores_discrete[k + '_val_acc'], scores_discrete[k + '_IoU'] = calculate_scores(cl, pred_cl)
        scores_discrete[k + '_val_acc'], scores_discrete[k + '_IoU'] = calculate_scores(cl, pred_cl)
        scores_discrete[k + '_val_acc'], scores_discrete[k + '_IoU'] = calculate_scores(cl, pred_cl)
        
    #### Regression
    regression_labels = ['relative_angle', 'center_distance', 'veh_distance']
    for k in regression_labels:
      scores_SMPE[k + '_mean_SMPE'] = smpe = np.mean(abs(preds[k] - labels[k])/((abs(labels[k]) + abs(preds[k]))/2))
      scores_MAE[k + '_mean_MAE'] = mae = np.mean(abs(labels[k] - preds[k]))
      
      scores_MAE_F1[k + '_mean_MAE_F1'] = F.l1_loss(torch.tensor(preds[k].squeeze(), requires_grad = True), torch.tensor(labels[k], requires_grad = True))
      
      mape = 0
      c = 0
      for i,j in zip(labels[k],preds[k]):
        if i==0.0:
          c = c + 1
          mape = mape + abs((i-j)/(i + 1))
        else:
          mape = mape + abs((i-j)/(i))
          #if k =='relative_angle':
           # print('label pred',i,j)
        scores_MAPE[k + '_mean_MAPE'] = mape/len(labels[k])
        #print(k)
        #print("c, k:", c, k)

    

    #print("The code is for 2275 key Images")
    #print("this code is running")
    
        #scores[k + '_MAE_mean'] = mae = np.mean(abs((labels[k] - preds[k])/(labels[k] + 10**(-18))))
    return scores_discrete, scores_MAE, scores_MAPE, scores_SMPE, scores_MAE_F1
