import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def data_generation(n, shape, scale, C_haz, end):
    T = scale*np.random.weibull(shape, n)
    ind = T > end
    T[ind] = np.inf

    #Add censoring
    C = np.random.exponential(C_haz,n)
    Cind = T>C
    T[Cind] = C[Cind]
    E = 1*(~Cind)   # E==1 if event happened before censoring
    return [T,E]

def weibS(x,shape,scale):
    return np.exp(-(x/scale)**shape)
def weibh(x, shape, scale):
    return (scale**(-shape))*shape*(x**(shape-1))
def weibH(x, shape, scale):
    return (x/scale)**shape

# Generate data
n = 1000
shape1 = 1
scale1 = 2
shape2 = 2
scale2 = 2**0.5

haz_cross = 0.5
end = 1.1
C_haz = 20

[T1,E1] = data_generation(n, shape1, scale1, C_haz, end)
[T2,E2] = data_generation(n, shape2, scale2, C_haz, end)

def haz_calc(t, x):
    if x == 0 and t < end:
        return weibh(t, shape1, scale1)
    elif x == 1 and t < end:
        return weibh(t, shape2, scale2)
    else:
        return 0
def haz_calc_wrong(t, x):
    if x == 0 and t < end:
        return weibh(t, shape1, scale1)
    elif x == 1 and t < haz_cross:
        return weibh(t, shape2, scale2)
    elif x == 1 and t >= haz_cross and t < end:
        return weibh(t, shape2, scale2)*10
    else:
        return 0
def haz_calc_wrong2(t,x):
    if x == 0 and t < end:
        return (1/2)*weibh(t, shape1, scale1)
    elif x == 1 and t < end:
        return weibh(t, shape2, scale2)
    else:
        return 0
def haz_calc_wrong3(t,x):
    if x == 0 and t < end:
        return weibh(t, shape1, scale1)
    elif x == 1 and t < end:
        return (1/2)*weibh(t, shape2, scale2)
    else:
        return 0


def cumhaz_calc(t, x):
    if x == 0:
        if t < end:
            return weibH(t, shape1, scale1)
        else:
            return weibH(end, shape1, scale1)
    else:
        if t < end:
            return weibH(t, shape2, scale2)
        else:
            return weibH(end, shape2, scale2)
def cumhaz_calc_wrong(t, x):
    if x == 0:
        if t < end:
            return weibH(t, shape1, scale1)
        else:
            return weibH(end, shape1, scale1)
    else:
        if t < haz_cross:
            return weibH(t, shape2, scale2)
        elif t < end:
            return weibH(haz_cross, shape2, scale2) + 10 * (weibH(t, shape2, scale2) - weibH(haz_cross, shape2, scale2))
        else:
            return weibH(haz_cross, shape2, scale2) + 10 * (weibH(end, shape2, scale2) - weibH(haz_cross, shape2, scale2))
def cumhaz_calc_wrong2(t, x):
    if x == 0:
        if t < end:
            return (1/2)*weibH(t, shape1, scale1)
        else:
            return (1/2)*weibH(end, shape1, scale1)
    else:
        if t < end:
            return weibH(t, shape2, scale2)
        else:
            return weibH(end, shape2, scale2)
def cumhaz_calc_wrong3(t, x):
    if x == 0:
        if t < end:
            return weibH(t, shape1, scale1)
        else:
            return weibH(end, shape1, scale1)
    else:
        if t < end:
            return (1/2)*weibH(t, shape2, scale2)
        else:
            return (1/2)*weibH(end, shape2, scale2)

#Plot
my_dpi = 120
plt.figure(figsize=(2400/my_dpi, 1000/my_dpi), dpi=my_dpi)

plt.subplot(2,4,1)
plt.plot([i/100 for i in range(1,int(end*100))], [weibh(i/100, shape1, scale1) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [weibh(i/100, shape2, scale2) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.ylabel(r'$\alpha (t\vert z)$')

plt.subplot(2, 4, 5) # index 2
plt.plot([i/100 for i in range(1,int(end*100))], [weibH(i/100, shape1, scale1) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [weibH(i/100, shape2, scale2) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.xlabel('t')
plt.ylabel(r'$H(t\vert z)$')

plt.subplot(2,4,2)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)

plt.subplot(2, 4, 6) # index 2
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.xlabel('t')

plt.subplot(2,4,3)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong2(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong2(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)

plt.subplot(2, 4, 7) # index 2
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong2(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong2(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.xlabel('t')

plt.subplot(2,4,4)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong3(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong3(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)

plt.subplot(2, 4, 8) # index 2
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong3(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong3(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.xlabel('t')

plt.savefig('p2.png', bbox_inches = 'tight' , dpi =my_dpi)



plt.subplot(2, 4, 5) # index 2
plt.plot([i/100 for i in range(1,int(end*100))], [weibH(i/100, shape1, scale1) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [weibH(i/100, shape2, scale2) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.xlabel('t')
plt.ylabel(r'$H(t\vert z)$')

plt.subplot(2,4,2)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)

plt.subplot(2, 4, 6) # index 2
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.xlabel('t')

plt.subplot(2,4,3)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong2(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong2(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)

plt.subplot(2, 4, 7) # index 2
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong2(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong2(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.xlabel('t')

plt.subplot(2,4,4)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong3(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [haz_calc_wrong3(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)

plt.subplot(2, 4, 8) # index 2
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong3(i/100, 0) for i in range(1,int(end*100))], '-k', alpha =0.5)
plt.plot([i/100 for i in range(1,int(end*100))], [cumhaz_calc_wrong3(i/100, 1) for i in range(1,int(end*100))],'--k', alpha = 1)
plt.xlabel('t')

plt.savefig('p3.png', bbox_inches = 'tight' , dpi =my_dpi)


