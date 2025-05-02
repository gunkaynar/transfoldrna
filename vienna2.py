
import RNA
from Bio.Seq import Seq
f = open('test_examples2.txt', 'r')
lines = f.readlines()
f.close()

#Input RNA: 
#True Structure: 
#Predicted Structure: 

dic = {}
for line in lines:
    if line.startswith('Input RNA'):
        rna_str = line.split(': ')[1].strip()
        dic[rna_str] = []
    elif line.startswith('True Structure'):
        true_str = line.split(': ')[1].strip()
        dic[rna_str].append(true_str)
    elif line.startswith('Predicted Structure'):
        pred_str = line.split(': ')[1].strip()
        dic[rna_str].append(pred_str)

def calculate_metrics(pred, true):
    """
    Calculate the sensitivity, specificity, and accuracy of the predicted structure
    compared to the true structure.

    Parameters
    ----------
    pred : str
        Predicted dot‑bracket notation ('.', '(', ')').
    true : str
        True dot‑bracket notation ('.', '(', ')').

    Returns
    -------
    tuple
        Sensitivity, specificity, and accuracy.
    """
    tp = sum(1 for p, t in zip(pred, true) if p == '(' and t == '(')
    tn = sum(1 for p, t in zip(pred, true) if p == '.' and t == '.')
    fp = sum(1 for p, t in zip(pred, true) if p == '(' and t == '.')
    fn = sum(1 for p, t in zip(pred, true) if p == '.' and t == '(')



    return tp, tn, fp, fn


#calculate tp, tn, fp, fn on the dic
all_metrics = []
for key in dic.keys():
    pred = dic[key][1]
    true = dic[key][0]
    tp, tn, fp, fn = calculate_metrics(pred, true)
    all_metrics.append((tp, tn, fp, fn))

# sum all metrics
tp_sum = sum([x[0] for x in all_metrics])
tn_sum = sum([x[1] for x in all_metrics])
fp_sum = sum([x[2] for x in all_metrics])
fn_sum = sum([x[3] for x in all_metrics])

# calculate sensitivity, specificity, and accuracy
sensitivity = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
specificity = tn_sum / (tn_sum + fp_sum) if (tn_sum + fp_sum) > 0 else 0
accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum) if (tp_sum + tn_sum + fp_sum + fn_sum) > 0 else 0
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")
#precision recall f1 aucroc aucpr
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
# Create true and predicted labels
y_true = []
y_pred = []
for key in dic.keys():
    pred = dic[key][1]
    true = dic[key][0]
    y_true.extend([1 if x == '(' else 0 for x in true])
    y_pred.extend([1 if x == '(' else 0 for x in pred])
# Calculate precision, recall, f1 score, roc auc, and pr auc
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")


#heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create a confusion matrix
confusion_matrix = np.array([[tp_sum, fp_sum],
                               [fn_sum, tn_sum]])
# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Positive', 'Predicted Negative'],
            yticklabels=['True Positive', 'True Negative'],annot_kws={"size": 20})

#font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

#plt.title('Confusion Matrix')

plt.savefig('confusion_matrix.png', dpi=300)
plt.clf()
