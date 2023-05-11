import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt

def triangular(t, x, y, b):
    xt = np.abs(x - t)
    xt = b - xt
    ind = xt > 0
    return np.sum(xt[ind] * y[ind]) / np.sum(xt[ind])

def smoothed_triangular(t, x, y, b):
    if t < b:
        ind2 = x < b
        x_extra = x[ind2]
        x_extra = -x_extra
        y_extra = y[ind2]
        y_extra = 1 + (1 - y_extra)
        x = np.append(x, x_extra)
        y = np.append(y, y_extra)
    xt = np.abs(x - t)
    xt = b - xt
    ind = xt > 0

    out = np.sum(xt[ind] * y[ind]) / np.sum(xt[ind])
    return out

def dsdt(t, x, y, b, c):
    return (smoothed_triangular(t + c/2, x, y, b) - smoothed_triangular(t-c/2, x, y, b)) / c

def haz_rate(t, x, y, b, c):
    return -dsdt(t, x, y, b, c) / smoothed_triangular(t, x, y, b)

def calc_concordance(m1, df_temp):
    n_counts = 0
    counts = 0
    n = len(df_temp)
    T = df_temp['t']
    E = df_temp['e']

    # Iterate through every indivdual pair of individuals once (so i<j)
    for i in range(n):
        for j in range(i + 1, n):
            # Determine which of the two had the first event
            index_sorted = np.argsort([T[i], T[j]])
            index_min = [i, j][index_sorted[0]]
            index_max = [i, j][index_sorted[1]]

            # Is the first observation an event? if so proceed. If not comparable pass.
            if E[index_min] == 1:
                n_counts += 1  # Add to comparable count

                # risk for individual 1
                h1 = m1[index_min, index_min]
                h2 = m1[index_min, index_max]

                h1 = float(h1)
                h2 = float(h2)
                if h1 == h2:
                    counts += 0.5
                if h1 > h2:
                    counts += 1
    print("Total comparable events: " + str(n_counts) + "\nTotal concordant events: " + str(counts)
          + "\nConcordance: " + str(counts / n_counts))
    return counts / n_counts

def get_hazards_matrix(t, e, g, end_t, b=0.25, c=0.01, plot_name='haz'):
    kmf1 = KaplanMeierFitter()
    kmf1.fit(t[g == 0], e[g == 0])
    kmf2 = KaplanMeierFitter()
    kmf2.fit(t[g == 1], e[g == 1])

    x1 = kmf1.timeline
    y1 = kmf1.survival_function_
    y1 = np.asarray(y1)
    y1.shape = x1.shape

    x2 = kmf2.timeline
    y2 = kmf2.survival_function_
    y2 = np.asarray(y2)
    y2.shape = x2.shape

    data_temp = {'t': t,
                 'e': e,
                 'g': g}
    df_temp = pd.DataFrame(data_temp)
    df_temp = df_temp.sort_values(by=['t'])

    i = sum(df_temp['t'] < end_t)
    df_temp['t'][i:] = end_t
    df_temp['e'][i:] = 0

    haz_rate_0 = [haz_rate(t, x1, y1, b, c) for t in df_temp['t']]
    haz_rate_1 = [haz_rate(t, x2, y2, b, c) for t in df_temp['t']]
    haz_matrix = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        if df_temp['g'].iloc[i] == 0:
            haz_matrix[:, i] = haz_rate_0
        else:
            haz_matrix[:, i] = haz_rate_1

    kmf1.plot(label=r'$\hat{S}(t\vert Z_{i}=0)$')
    kmf2.plot(label=r'$\hat{S}(t\vert Z_{i}=1)$')
    plt.xlim(0,end_t)
    plt.xlabel('t')
    plt.savefig('KaplanMeier/' + plot_name + '_1.pdf')
    plt.close()

    plt.plot(df_temp['t'], haz_rate_0)
    plt.plot(df_temp['t'], haz_rate_1)
    plt.legend([r'$\hat{\alpha}(t\vert Z_{i}=0)$', r'$\hat{\alpha}(t\vert Z_{i}=1)$'])
    plt.xlabel('t')
    plt.savefig('KaplanMeier/' + plot_name + '_2.pdf')
    plt.close()
    return haz_matrix, df_temp

def get_neg_surv_matrix(t, e, g, end_t, plot_name='surv'):
    kmf1 = KaplanMeierFitter()
    kmf1.fit(t[g == 0], e[g == 0])
    kmf2 = KaplanMeierFitter()
    kmf2.fit(t[g == 1], e[g == 1])

    data_temp = {'t': t,
                 'e': e,
                 'g': g}
    df_temp = pd.DataFrame(data_temp)
    df_temp = df_temp.sort_values(by=['t'])

    i = sum(df_temp['t'] < end_t)
    df_temp['t'][i:] = end_t
    df_temp['e'][i:] = 0

    surv0 = 1 - kmf1.survival_function_at_times(df_temp['t'])
    surv1 = 1 - kmf2.survival_function_at_times(df_temp['t'])

    plt.plot(df_temp['t'], surv0, label = 'Z=0')
    plt.plot(df_temp['t'], surv1, label = 'Z=1')
    plt.savefig('KaplanMeier/' + plot_name + '.png')
    plt.close()

    neg_surv_matrix = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        if df_temp['g'].iloc[i] == 0:
            neg_surv_matrix[:, i] = surv0
        else:
            neg_surv_matrix[:, i] = surv1
    return neg_surv_matrix, df_temp

def get_neg_surv_t_matrix(t, e, g, end_t,  t_eval, plot_name= 'surv_t'):
    kmf1 = KaplanMeierFitter()
    kmf1.fit(t[g == 0], e[g == 0])
    kmf2 = KaplanMeierFitter()
    kmf2.fit(t[g == 1], e[g == 1])

    S0 = np.float32(kmf1.survival_function_at_times(t_eval))
    S1 = np.float32(kmf2.survival_function_at_times(t_eval))

    data_temp = {'t': t,
                 'e': e,
                 'g': g}
    df_temp = pd.DataFrame(data_temp)
    df_temp = df_temp.sort_values(by=['t'])

    i = sum(df_temp['t'] < end_t)
    df_temp['t'][i:] = end_t
    df_temp['e'][i:] = 0

    surv0 = 1-S0*np.ones(len(t))
    surv1 = 1-S1*np.ones(len(t))


    neg_surv_matrix = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        if df_temp['g'].iloc[i] == 0:
            neg_surv_matrix[:, i] = surv0
        else:
            neg_surv_matrix[:, i] = surv1
    return neg_surv_matrix, df_temp

def get_quant_t_matrix(t, e, g, end_t,  q_eval, plot_name= 'surv_t'):
    kmf1 = KaplanMeierFitter()
    kmf1.fit(t[g == 0], e[g == 0])
    kmf2 = KaplanMeierFitter()
    kmf2.fit(t[g == 1], e[g == 1])

    qcount0 = sum(np.asarray(kmf1.survival_function_)>q_eval)
    qt0 = kmf1.timeline[qcount0]
    qcount1 = sum(np.asarray(kmf2.survival_function_)>q_eval)
    qt1 = kmf2.timeline[qcount1]

    data_temp = {'t': t,
                 'e': e,
                 'g': g}
    df_temp = pd.DataFrame(data_temp)
    df_temp = df_temp.sort_values(by=['t'])

    i = sum(df_temp['t'] < end_t)
    df_temp['t'][i:] = end_t
    df_temp['e'][i:] = 0

    surv0 = 1-qt0*np.ones(len(t))
    surv1 = 1-qt1*np.ones(len(t))


    neg_surv_matrix = np.zeros((len(t), len(t)))
    for i in range(len(t)):
        if df_temp['g'].iloc[i] == 0:
            neg_surv_matrix[:, i] = surv0
        else:
            neg_surv_matrix[:, i] = surv1
    return neg_surv_matrix, df_temp

# Make a function to run a test
def run_test(n0, n1, t_break, h0, h1, h2, h3, b, c, end_time1, end_time2, experiment_name):
    n = n0 + n1
    group_1 = np.zeros(n)
    group_1[n0:] = 1

    X0 = (1 / h0) * np.random.weibull(1, n0)
    X1 = (1 / h2) * np.random.weibull(1, n1)

    X0[X0 > t_break] = t_break + (1 / h1)*np.random.weibull(1, sum(X0 > t_break))
    X1[X1 > t_break] = t_break + (1 / h3) * np.random.weibull(1, sum(X1 > t_break))

    T0 = [min(X0[i], end_time1) for i in range(len(X0))]
    T1 = [min(X1[i], end_time1) for i in range(len(X1))]

    E0 = X0 < end_time1
    E1 = X1 < end_time1

    T = np.zeros(n)
    T[group_1 == 0] = T0
    T[group_1 == 1] = T1

    E = np.zeros(n)
    E[group_1 == 0] = E0
    E[group_1 == 1] = E1

    data = {'T': T,
            'E': E,
            'g1': group_1}

    df = pd.DataFrame(data)
    haz_matrix1, df_temp = get_hazards_matrix(df['T'], df['E'], df['g1'], end_time2, b=b, c=c, plot_name='haz_'+experiment_name)
    calc_concordance(haz_matrix1, df_temp)

    neg_surv_matrix1, df_temp = get_neg_surv_matrix(df['T'], df['E'], df['g1'], end_time2, plot_name='surv_'+experiment_name)
    calc_concordance(neg_surv_matrix1, df_temp)

    neg_surv_t_matrix, df_temp = get_neg_surv_t_matrix(df['T'], df['E'], df['g1'], end_time2, 0.5, plot_name='surv_'+experiment_name)
    calc_concordance(neg_surv_t_matrix, df_temp)

    neg_q_matrix, df_temp = get_quant_t_matrix(df['T'], df['E'], df['g1'], end_time2, 0.25, plot_name='surv_'+experiment_name)
    calc_concordance(neg_q_matrix, df_temp)

    neg_q_matrix, df_temp = get_quant_t_matrix(df['T'], df['E'], df['g1'], end_time2, 0.5, plot_name='surv_'+experiment_name)
    calc_concordance(neg_q_matrix, df_temp)

    neg_q_matrix, df_temp = get_quant_t_matrix(df['T'], df['E'], df['g1'], end_time2, 0.75, plot_name='surv_'+experiment_name)
    calc_concordance(neg_q_matrix, df_temp)

    return

# b,c >0, c<b/2
run_test(2000, 2000, 0.1, 6, 1, 1.4, 1.4, b= 0.05, c = 0.025, end_time1 = 1.1, end_time2= 1, experiment_name='test_1')
run_test(2000, 2000, 0.9, 0.5, 10, 2, 1, b= 0.05, c = 0.025, end_time1 = 1.1, end_time2= 1, experiment_name='test_2')

