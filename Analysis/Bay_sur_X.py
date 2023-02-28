import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# calc Baysian surprise
def bayes_surp(prob):
    bays = []
    for i in range(1,prob.shape[0]):
        s = 0
        for j in range(1,i):
            s -= prob[j] * math.log(prob[j-1])
            
            # s -= prob[j] * math.log(prob[j-1]/prob[j])
        bays.append(s)
    return np.array(bays)
    

# calc the derivative of the bayesian surprise
def bayes_der(arr):
    output = []
    for i in range(1,arr.shape[0]):
        output.append(arr[i]-arr[i-1])
    output = np.array(output)
    return output

# Measure transient increase in disfluency
# smoothing parameter for the running average
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
    

# smooth the curve
def smooth(arr,win):
    output = []
    for i in range(win,arr.shape[0]):
        p = 0
        for j in range(i-win,i):
            p += arr[j]
        p /= win
        output.append(p)
    output = np.array(output)
    return output

# convert the discrete signal into wave
def wave(arr):
    avg = np.average(arr)
    adj = np.ones(arr.shape[0]) * avg
    arr -= adj
    wav = np.zeros(arr.shape[0])
    wav[0] = arr[0]
    for i, amp in enumerate(arr):
        if i == 0: continue
        wav[i] = amp + wav[i-1]
    return wav


# visualization
def plot(arr):
    x = np.arange(arr.shape[0])
    plt.plot(x,arr)

if __name__ == '__main__':
    # load the data
    # path = 'Results/Tunnel_result.csv'
    # path = 'Results/Montuary_result.csv'
    path = 'Results/Little_Match_Girl_result.csv'
    path = 'Results/Last_Leaf_result.csv'
    data = pd.read_csv(path,delimiter='/')
    prob = np.array(data[' prob'])

    # Derive the bayesian surprise from the probs
    bays = bayes_surp(prob)

    # derivative, and dif visualization
    dif = bayes_der(bays)
    smooth_prob = smooth(dif,1)
    wave_prob = wave(dif)

    # Transient increase in disfluency (same as paper)
    drift = 0.03
    tran = tran_sur(bays,drift)

    plot(bays)
    print(tran.shape)