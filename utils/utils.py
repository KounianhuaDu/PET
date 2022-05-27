from sklearn.metrics import roc_auc_score, log_loss

def evaluate_auc(pred, label):
    res=roc_auc_score(y_score=pred, y_true=label)
    return res

def evaluate_logloss(pred, label):
    res = log_loss(y_true=label, y_pred=pred,eps=1e-7, normalize=True)
    return res
    
