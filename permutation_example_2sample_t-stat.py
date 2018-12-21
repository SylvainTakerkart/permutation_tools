import numpy as np
from numpy.random import random, permutation
from scipy.stats import ttest_ind


#####################################
#
# two-sample t-test: permuting labels
#
# null hypothesis: there is no difference between the mean of the two groups of observations / values
#
#####################################

n_samples_per_group = 50

# define signal to be added to only the examples of one class
mu_signal = [0,1]
X_signal = np.hstack([mu_signal[0]*np.ones(n_samples_per_group),mu_signal[1]*np.ones(n_samples_per_group) ])

# define background noise for all examples
noise_size = 10
X_noise = noise_size * np.random.random(2*n_samples_per_group)

# define noisy observations
X = X_signal + X_noise

# define the label of each observation, for each of the two groups (respectively 0 and 1)
y = np.hstack([np.zeros(n_samples_per_group),np.ones(n_samples_per_group)])

# define permutation indices (the first set of indices is the real one)
n_permuts = 1000
permut_inds = np.zeros([n_permuts,len(y)],dtype=np.int)
permut_inds[0,:] = np.arange(len(y),dtype=np.int)
for perm_ind in range(n_permuts-1):
    permut_inds[perm_ind+1,:] = np.random.permutation(len(y))

# compute t-score for each set of permuted group definitions
scores_list = []
for perm_ind in range(n_permuts):
    group1_data = X[y[permut_inds[perm_ind,:]]==1]
    group2_data = X[y[permut_inds[perm_ind,:]]==0]
    t_res = ttest_ind(group1_data,group2_data)
    scores_list.append(t_res.statistic)

# computing p-value (that the null hypothesis of no difference between groups is true)
scores_list = np.array(scores_list)
true_score = scores_list[0]
p_val = sum(true_score<=scores_list) / float(n_permuts)

print("The true t-score is {:f}".format(true_score))
print("The probability that the null hypothesis of no difference groups classes is valid is {:f}".format(p_val))


