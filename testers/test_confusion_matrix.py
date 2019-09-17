from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

data = pd.read_csv("/home/anapt/export_dataframe.csv")

confusion_mat = confusion_matrix(data['true_label'], data['predicted_label'])

labels = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]

plt.figure(figsize=(16, 9))
seaborn.heatmap(confusion_mat, cmap="Blues", annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix", fontsize=20)
plt.xlabel('Predicted Class', fontsize=10)
plt.ylabel('Original Class', fontsize=10)
plt.tick_params(labelsize=7)
plt.xticks(rotation=0)
plt.yticks(rotation=90)
plt.savefig("./conf_boot1.pdf")
# plt.show()
