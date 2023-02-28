import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt
from IPython.display import display
from Bay_sur_X import plot, wave
plt.rcParams['text.usetex'] = True

def load(path):
    word = []
    dist = []
    with open(path,'r') as f:
        for i,row in enumerate(f):
            if i % 2 == 0:
                word.append(row[:-1])
            else:
                dist.append(np.fromstring(row,sep=','))
    return word,np.array(dist)

# calc bayesian surprise based on the formula provided
def bay_sur(pv : np.array):
    out = []
    for i in range(1,pv.shape[0]):
        arr = np.log(pv[i-1,:])
        out.append(-np.dot(pv[i,:],arr))
    return np.array(out)

# calc entropy
def ent(pv : np.array):
    out = []
    for i in range(pv.shape[0]):
        out.append(-np.dot(pv[i,:],np.log(pv[i,:])))
    return np.array(out)

# calc the transient surprise
def tran_sur(arr, drift):
    u = [np.average(arr)]
    output = [arr[0]/u[0]]
    for i in range(1,arr.shape[0]):
        u.append(
            u[i-1] + (arr[i]-u[i-1]) * drift
            )
        output.append(
            arr[i]/u[i-1]
            )
    return np.array(output)


if __name__ == '__main__':
    # path = 'last_leaf'
    path = 'little_match_girl'
    # path = 'mortuary'
    word,pv = load(f'Results_Dist/{path}_result.csv')
    # sav = dict()
    # sav['word'] = word

    # bs = bay_sur(pv)
    # en = ent(pv)
    # sav['bs'] = bs
    # sav['entropy'] = en
    # with open(f'Bayesian_Surprise_Entropy/{path}.pickle','wb') as f:
    #     pickle.dump(sav,f)


    with open(f'Bayesian_Surprise_Entropy/{path}.pickle','rb') as f:
        dict = pickle.load(f)
        bs = dict['bs']
        word = dict['word']
        en = dict['entropy']
        del dict


    ts = tran_sur(bs,0.05)
    ind = np.argmax(bs)+1
    max_indices = np.argsort(ts)[-5:]
    for i in max_indices:
        p1 = pv[i+1]
        p2 = pv[i]
        print(i+1,word[i+1])
        print(i,word[i])

        sep = ''
        plt.figure(1)
        # plt.title(fr'{sep.join(word[i-6:i])} $\bf{word[i]}$')
        plt.title(fr'{sep.join(word[i-6:i])}{word[i]}')
        plt.ylim(0,1)
        plt.plot(p2)
        plt.figure(2)
        # plt.title(fr'{sep.join(word[i-6:i+1])} $\bf{word[i+1]}$')
        plt.title(fr'{sep.join(word[i-6:i+1])}{word[i+1]}')
        plt.ylim(0,1)
        plt.plot(p1)
        plt.show()
    # ts = wave(bs)
    # plt.figure(1)
    # plot(bs)
    # plt.figure(2)
    # plot(ts)
    # plt.figure(3)
    # plot(en)
    plt.show()

