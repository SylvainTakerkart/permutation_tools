import numpy as np
from numpy.random import random, permutation
from scipy.stats import ttest_1samp


#####################################
#
# one-sample t-test: permuting signs of observations
#
# null hypothesis: the mean of the observed samples is zero
#
#####################################

n_samples = 20

# define random observations
mu_signal = 0.2
noise_size = 0.5
X = np.random.normal(loc=mu_signal, scale=noise_size, size=n_samples)

# define permutation of signs (the first set of indices is the real one, i.e all positive signs)
n_permuts = 1000
permuted_signs = np.zeros([n_permuts,n_samples],dtype=np.int)
permuted_signs[0,:] = np.ones(n_samples)
for perm_ind in range(n_permuts-1):
    permuted_signs[perm_ind+1,:] = np.random.randint(2, size=n_samples) * 2 - 1

# compute t-score for each set of permuted group definitions
scores_list = []
for perm_ind in range(n_permuts):
    X_permuted = np.multiply(X,permuted_signs[perm_ind,:])
    t_res = ttest_1samp(X_permuted, popmean=0)
    scores_list.append(t_res.statistic)

# computing p-value (that the null hypothesis of no difference between groups is true)
scores_list = np.array(scores_list)
true_score = scores_list[0]
p_val = sum(true_score<=scores_list) / float(n_permuts)

print("The true t-score is {:f}".format(true_score))
print("The probability that the null hypothesis is valid is {:f}".format(p_val))


