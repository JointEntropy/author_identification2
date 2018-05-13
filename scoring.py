from dataset import inverse_ohe
import pandas as pd
from scipy.stats import mode
from sklearn.metrics import f1_score
import numpy as np


def f2_grouped_score(preds, targets, group_labels, ohe):
    # группируем предсказанные метки для одного и тоже произведения и усредняем их
    pred_df = pd.DataFrame({ 'preds': inverse_ohe(preds, ohe),
                            'targets':inverse_ohe(targets, ohe),
                            'groups': group_labels})
    # группируем ground truth метки для одного и тоже произведения
    y_pred = pred_df.groupby('groups')['preds'].apply(list).apply(lambda x: mode(x)[0][0]).values
    y_target = pred_df.groupby('groups')['targets'].first().values

    score_by_classes = f1_score(y_target, y_pred, average=None)
    score = np.median(score_by_classes)
    return score
