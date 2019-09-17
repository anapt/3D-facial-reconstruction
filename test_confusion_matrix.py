from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn
import pathlib
import tensorflow as tf
import pandas as pd
import numpy as np

y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

data = pd.read_csv("/home/anapt/export_dataframe.csv")

confusion_mat = confusion_matrix(data['true_label'], data['predicted_label'])

labels = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]

plt.figure(figsize=(16, 7))
seaborn.heatmap(confusion_mat, cmap="Blues", annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix", fontsize=30)
plt.xlabel('Predicted Class', fontsize=20)
plt.ylabel('Original Class', fontsize=20)
plt.tick_params(labelsize=15)
plt.xticks(rotation=90)
plt.show()
