import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt

n = 10000
h1 = 0.5
h2 = 0.05

#calculate the pmf and then the cdf?
hazard1 = [h1 for i in range(5)]
hazard1.extend([h2 for i in range(5)])

hazard2 = [h2 for i in range(5)]
hazard2.extend([h1 for i in range(5)])

pi1 = [hazard1[0]]
for i in range(1, 10):
        pi1.append(hazard1[i] * (1- np.sum(pi1)))
pi2 = [hazard2[0]]
for i in range(1, 10):
        pi2.append(hazard2[i] * (1 - np.sum(pi2)))

F1 = np.cumsum(pi1)
F1 = np.append(F1, F1[-1]+(1 - F1[-1])/2)
F1 = np.append(F1, 1)

F2 = np.cumsum(pi2)
F2 = np.append(F2, F2[-1]+(1 - F2[-1])/2)
F2 = np.append(F2, 1)


#H1 = np.cumsum(hazard1)
#S1 = np.exp(-H1)
#F1 = 1 - S1
#F1 = np.append(F1, F1[-1]+(1 - F1[-1])/2)
#F1 = np.append(F1, 1)
#
#H2 = np.cumsum(hazard2)
#S2 = np.exp(-H2)
#F2 = 1 - S2
#F2 = np.append(F2, F2[-1]+(1 - F2[-1])/2)
#F2 = np.append(F2, 1)

# discrete hazard rate for the first 5 seconds (increments) is 0.5.
Ft1 = np.random.uniform(0,1,n)
Ft2 = np.random.uniform(0,1,n)

T1 = np.zeros_like(Ft1)
E1 = np.ones_like(Ft1)
for ind, i in enumerate(Ft1):
        T1[ind] = np.sum(i > F1)
        if T1[ind] >= 10:
                E1[ind] = 0
                T1[ind] = 10

T2 = np.zeros_like(Ft2)
E2 = np.ones_like(Ft2)
for ind, i in enumerate(Ft2):
        T2[ind] = np.sum(i > F2)
        if T2[ind] >= 10:
                E2[ind] = 0
                T2[ind] = 10

T = np.append(T1, T2)
E = np.append(E1, E2)
G = np.ones(2*n)
G[:n] = 0.

shuffle_index = np.random.choice(len(T), len(T))
T = T[shuffle_index]
E = E[shuffle_index]
G = G[shuffle_index]

data = {'T': T,
        'E': E,
        'G1': G}

df1 = pd.DataFrame(data)

# Should probably add several random noise variables? Categorical or otherwise
noise = {'G'+str(i): np.random.binomial(1, 0.5, 2*n) for i in range(2,10)}
df2 = pd.DataFrame(noise)

df = pd.concat((df1, df2), axis = 1)
df = df.astype('float32')

df.to_csv('sample data/SYNTHETIC_CROSSING/synthetic_crossing.csv', index = False)
#Alright now we want to train a neural network on this data. See if it finds the correct pattern... or not!

#kmf = KaplanMeierFitter()
#kmf.fit(T1, E1)  # censoring prob = survival probability of event "censoring"
#kmf.plot()
#
#kmf = KaplanMeierFitter()
#kmf.fit(T2, E2)  # censoring prob = survival probability of event "censoring"
#kmf.plot()






