import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from .utils import eval_step, create_batches


def evaluate_model(state, X, y, batch_size=32):
    """
    Evaluate the model on test set
    """
    batches = create_batches(X, y, batch_size, shuffle=False)

    all_preds = []
    for batch_x, _ in batches:
        preds = eval_step(state, batch_x)
        all_preds.extend(np.array(preds))

    all_preds = np.array(all_preds)
    pred_labels = (all_preds > 0.5).astype(int)

    accuracy = accuracy_score(y, pred_labels)
    f1 = f1_score(y, pred_labels, average='macro')

    # Extra metrics to check model performance
    conf_matrix = confusion_matrix(y, pred_labels)
    report = classification_report(y, pred_labels, target_names=['Negative', 'Positive'])

    return accuracy, f1, pred_labels, conf_matrix, report
