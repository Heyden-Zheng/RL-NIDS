from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true, y_pred):
    """计算分类评估指标"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted')
    }
    return metrics