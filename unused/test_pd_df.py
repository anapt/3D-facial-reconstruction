import numpy as np
import pandas as pd

data_resnet = pd.read_csv("./RESULTS/{}_loss.csv".format('resnet50'))
# print(data_resnet.head())
data_xception = pd.read_csv("./RESULTS/{}_loss.csv".format('xception'))
# print(data_xception.head())

xception = data_xception.values
resnet = data_resnet.values

# data_resnet.join(data_xception, lsuffix='_caller', rsuffix='_other')
# print(data_resnet.head())
data_resnet.insert(1, "xception_loss", xception, True)
data_xception.insert(1, "resnet_loss", resnet, True)

print(data_xception.to_latex(index=False))