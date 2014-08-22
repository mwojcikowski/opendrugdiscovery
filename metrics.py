from math import ceil
import numpy as np
from sklearn.metrics import roc_curve as roc, roc_auc_score as roc_auc, auc

log_min=0.001
log_max=1.
random_logauc = (log_max-log_min)/(np.log(10)*np.log10(log_max/log_min))

def enrichment_factor(y_true, y_score, percentage=1, pos_label=None):
    if pos_label is None:
        pos_label = 1
    labels = y_true == pos_label
    # calculate fraction of positve labels
    n_perc = ceil(float(percentage)/100.*len(labels))
    return labels[:n_perc].sum()/n_perc*100
    
def roc_log_auc(y_true, y_score, pos_label=None, log_min=0.001, log_max=1.):
    fpr, tpr, t = roc(y_true, y_score, pos_label=pos_label)
    idx = (fpr >= log_min) & (fpr <= log_max)
    log_fpr = 1-np.log10(fpr[idx])/np.log10(log_min)
    return auc(log_fpr, tpr[idx])
