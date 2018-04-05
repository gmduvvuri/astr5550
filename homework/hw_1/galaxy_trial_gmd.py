# Importing standard numerical + plotting libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
np.random.seed(123)
sns.set_context('paper')


# For a given probability, repeat experiment
# until desired outcome, then return number of attempts
def find_num_trials(prob=0.3):
    observed = False
    num_trials = 0
    while not observed:
        if np.random.uniform() <= prob:
            observed = True
        num_trials += 1
    return num_trials


# For a given number of trials and probability,
# use find_num_trials() to get a number of attempts
# for each trial and return an array with these numbers
def run_n_attempts(n=1000, prob=0.3):
    num_trials_array = np.zeros((n))
    for i in range(0, len(num_trials_array)):
        num_trials_array[i] = find_num_trials(prob)
    return num_trials_array


# Find the P[k] function arrays for limits on k
def plot_func(num_trials_array, prob=0.3):
    min_num = np.min(num_trials_array)
    max_num = np.max(num_trials_array)
    x_array = np.arange(min_num, max_num+1, 1)
    y_array = (prob)*((1-prob)**(x_array - 1))
    return x_array, y_array*len(num_trials_array)


# Plot the frequency distribution for given N, probability
# plot the N*P[k] expression for comparison
def plot_histogram(num_trials_array, prob=0.3, bin_num=20):
    plt.figure(figsize=(6*1.3, 4.5*1.3))
    plt.hist(num_trials_array, bins=bin_num, label='Trials')
    func_x, func_y = plot_func(num_trials_array, prob)
    func_name = r'$N \times P[k] = N \times '
    func_name += str(1.0 - prob) + r'^{k-1} \times '
    func_name += str(prob) + r'$'
    plt.plot(func_x, func_y, '-k', label=func_name)
    title_name = r'Frequency Distribution of Number of Images'
    title_name += ' Required to Find Galaxy Cluster'
    title_name += r' ($N=' + str(len(num_trials_array)) + ')$'
    plt.suptitle(title_name)
    plt.xlabel(
        r'$k$ [Number of Images required before finding Galaxy Cluster]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('hist.pdf')
    plt.show()


if __name__ == '__main__':
    num_trials_array = run_n_attempts(1000)
    plot_histogram(num_trials_array)
