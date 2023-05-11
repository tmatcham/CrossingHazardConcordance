import numpy as np
from scipy.optimize import minimize
import csv

def data_generation(n, shape, scale, C_haz, end):
    T = scale*np.random.weibull(shape, n)
    ind = T > end
    T[ind] = np.inf

    #Add censoring
    C = np.random.exponential(C_haz,n)
    C[C>end] = end
    Cind = T>C
    T[Cind] = C[Cind]
    E = 1*(~Cind)   # E==1 if event happened before censoring
    return [T,E]
def calc_concordance(vec1, vec2,n, T, E):
    n_counts = 0
    counts = 0

    #Iterate through every indivdual pair of individuals once (so i<j)
    for i in range(2 * n):
        for j in range(i + 1, 2 * n):
            #Determine which of the two had the first event
            index_sorted = np.argsort([T[i], T[j]])
            index_min = [i, j][index_sorted[0]]
            index_max = [i, j][index_sorted[1]]

            # Is the first observation an event? if so proceed. If not comparable pass.
            if E[index_min] == 1:
                n_counts += 1 # Add to comparable count

                X1 = X[index_min]  # 1 represents the individual whos event/censoring is first
                X2 = X[index_max]

                # risk for individual 1
                if X1 == 0:
                    h1 = vec1[index_min]
                else:
                    h1 = vec2[index_min]

                # risk for individual 2
                if X2 == 0:
                    h2 = vec1[index_min]
                else:
                    h2 = vec2[index_min]

                h1 = float(h1)
                h2 = float(h2)
                if h1 == h2:
                    counts += 0.5
                if h1 > h2:
                    counts += 1
    print("Total comparable events: " + str(n_counts) + "\nTotal concordant events: " + str(counts)
          + "\nConcordance: " + str(counts/n_counts))
    return  counts / n_counts

def weibS(x,shape,scale):
    return np.exp(-(x/scale)**shape)
def weibh(x, shape, scale):
    return (scale**(-shape))*shape*(x**(shape-1))
def weibH(x, shape, scale):
    return (x/scale)**shape

n = 1000
shape1 = 1
scale1 = 2
shape2 = 2
scale2 = 2**0.5

haz_cross = 0.5

#Covariate
X = np.zeros((n))
X = np.append(X, np.ones((n)))

end = 1.1
C_haz = 20

def haz_calc(t, x):
    if x == 0:
        return weibh(t, shape1, scale1)
    elif x == 1:
        return weibh(t, shape2, scale2)
    else:
        return 0
def haz_calc_wrong(t, x):
    if x == 0:
        return weibh(t, shape1, scale1)
    elif x == 1 and t < haz_cross:
        return weibh(t, shape2, scale2)
    elif x == 1 and t >= haz_cross:
        return weibh(t, shape2, scale2)*10
    else:
        return 0
def haz_calc_wrong2(t,x):
    if x == 0:
        return (1/2)*weibh(t, shape1, scale1)
    elif x == 1:
        return weibh(t, shape2, scale2)
    else:
        return 0
def haz_calc_wrong3(t,x):
    if x == 0:
        return weibh(t, shape1, scale1)
    elif x == 1:
        return (1/2)*weibh(t, shape2, scale2)
    else:
        return 0


def cumhaz_calc(t, x):
    if x == 0:
        return weibH(t, shape1, scale1)
    else:
        return weibH(t, shape2, scale2)
def cumhaz_calc_wrong(t, x):
    if x == 0:
        return weibH(t, shape1, scale1)
    else:
        if t < haz_cross:
            return weibH(t, shape2, scale2)
        else:
            return weibH(haz_cross, shape2, scale2) + 10 * (weibH(t, shape2, scale2) - weibH(haz_cross, shape2, scale2))
def cumhaz_calc_wrong2(t, x):
    if x == 0:
        return (1/2)*weibH(t, shape1, scale1)
    else:
        return weibH(t, shape2, scale2)
def cumhaz_calc_wrong3(t, x):
    if x == 0:
        return weibH(t, shape1, scale1)
    else:
        return (1/2)*weibH(t, shape2, scale2)

def risk_score1(T, x, func):
    return [func(T[i], x) for i in range(len(T))]
def risk_score2(T, x, q, func):
    def find_median(t):
        return abs(func(t,x)-(-np.log(q)))
    median = minimize(find_median, 0.7, method='nelder-mead')
    return [-median.x[0] for i in range(len(T))]

function_list1 = [haz_calc, haz_calc_wrong, haz_calc_wrong2, haz_calc_wrong3,
                  cumhaz_calc, cumhaz_calc_wrong, cumhaz_calc_wrong2, cumhaz_calc_wrong3]

function_list2 = [cumhaz_calc, cumhaz_calc_wrong, cumhaz_calc_wrong2, cumhaz_calc_wrong3]

for iter in range(100):
    print('ITERATION   '+ str(iter))

    results_array = np.zeros((6, 4))

    [T1,E1] = data_generation(n, shape1, scale1, C_haz, end)
    [T2,E2] = data_generation(n, shape2, scale2, C_haz, end)

    T = np.append(T1, T2)
    E = np.append(E1, E2)

    #First two concordances - hazard and cumulative hazard at time of first event (Note cumulative hazard function is
    #one to one with negative survival.
    for ind,func in enumerate(function_list1):
        j = ind % 4
        i = ind // 4
        v1 = risk_score1(T, 0, func)       #risk scores for group 1
        v2 = risk_score1(T, 1, func)       #risk scores for group 2
        results_array[i,j] = calc_concordance(v1, v2, n =n , T= T, E = E)

    #Third and fourth concordance - cumulative hazard at fixed times
    for ind1,t in enumerate([0.5, 1.05]):
        for ind2, func in enumerate(function_list2):
            i = 2+ind1
            j = ind2
            v1 = risk_score1([t for i in range(len(T))], 0, func)
            v2 = risk_score1([t for i in range(len(T))], 1, func)
            results_array[i, j] = calc_concordance(v1, v2, n=n, T=T, E=E)

    #Fifth concordance - median predicted survival time
    for i,q in enumerate([0.5, 0.75]):
        for j, func in enumerate(function_list2):
            v1 = risk_score2(T, 0, q, func)
            v2 = risk_score2(T, 1, q, func)
            print(v1[0])
            print(v2[0])
            results_array[4+i,j] = calc_concordance(v1, v2, n =n , T= T, E = E)

    results_array.shape = (1,24)
    results_list = [i for i in results_array[0]]
    with open('record.csv', 'a', newline ='\n') as fp:
        fpr = csv.writer(fp)
        fpr.writerow(results_list)



with open('record.csv') as fp:
    rdr = csv.reader(fp)
    rows = [np.float32(i) for i in rdr]

data = np.asarray(rows)
avg = data.mean(axis = 0)
avg.shape = (6,4)
print(avg)

for j in range(6):
    counts = [0 for l in range(4)]
    for i in range(data.shape[0]):
        row = data[i,]
        row.shape = (6,4)
        max = np.max(row[j,])
        selection = [k for k in range(4) if row[j,k]==max]
        for k in selection:
            if k==0:
                print(i)
            counts[k] += 1
    print(counts)

