import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


data = pd.read_csv('result.csv')
print(data)

sur = 0
exp = []

for i in tqdm(range(3,data.shape[0])):
    # if data.loc[i,' hit'] == ' True':
    #     sur -= 0.1
    # else:
    #     sur += 0.1
    # sur = -math.log(100 * data.loc[i,' prob'])
    sur = (math.log(100 * data.loc[i-3,' prob']) + math.log(100 * data.loc[i-2,' prob'])-math.log(100 * data.loc[i-1,' prob']) - math.log(100 * data.loc[i,' prob']))/4
    exp.append([i, sur])

exp = np.array(exp)
# print(exp)

plt.plot(exp[:,0],exp[:,1])
plt.show()

