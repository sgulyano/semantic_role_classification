# Sklearn Report
import sklearn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from itertools import chain

def ner_classification_report(y_true, y_pred):
 
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    tagset = list(sorted(set(lb.classes_)))
    for v in ['pad', 'Z-O', 'Verb']:
        if v in tagset:
            tagset.remove(v)
    print(tagset)
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
        digits=4
    )