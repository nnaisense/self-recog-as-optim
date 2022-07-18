""" Run statistical analysis on experiment results
Usage:
    python run/analysis.py [SELF_RECOGNITION_DIRECTORY] [SEPARATE_POLICY_DIRCTORY] [EXPERIMENT_REPEATS]
Where:
    [SELF_RECOGNITION_DIRECTORY] is the directory created when you ran experiments with [SEPARATE_POLICIES] as false e.g. evolved self-recognising policies
    [SEPARATE_POLICY_DIRCTORY] is the directory created when you ran experiments with [SEPARATE_POLICIES] as True e.g. evolved separate policies
    [EXPERIMENT_REPEATS] is a positive integer indicating the number of evolutionary experiments that were ran
"""
import sys

import numpy as np
import torch

if __name__ == "__main__":

    self_rec_dir = sys.argv[1]
    seperate_dir = sys.argv[2]
    num_repeats = int(sys.argv[3])

    # Create buffers to track data
    all_self_train = []
    all_self_test = []
    all_seperate_train = []
    all_seperate_test = []

    # Iterate over runs
    for repeat in range(num_repeats):

        self_rec_path = f"{self_rec_dir}/{repeat}/data.np.npy"
        seperate_path = f"{seperate_dir}/{repeat}/data.np.npy"

        # Load the data for self-recognition
        self_data = np.load(self_rec_path, allow_pickle=True).item()
        # Get the train and test trajectories
        self_train = torch.tensor(self_data["train"]).cpu().numpy()
        self_test = torch.tensor(self_data["test"]).cpu().numpy()

        # Append train and test trajectories
        all_self_train.append(self_train)
        all_self_test.append(self_test)

        # Load the data for separate policies
        seperate_data = np.load(seperate_path, allow_pickle=True).item()
        # Get the train and test trajectories
        seperate_train = torch.tensor(seperate_data["train"]).cpu().numpy()
        seperate_test = torch.tensor(seperate_data["test"]).cpu().numpy()

        # Append train and test trajectories
        all_seperate_train.append(seperate_train)
        all_seperate_test.append(seperate_test)

    # Stack into [num_runs, num_generations] arrays
    all_self_test = np.stack(all_self_test)
    all_self_train = np.stack(all_self_train)
    all_seperate_train = np.stack(all_seperate_train)
    all_seperate_test = np.stack(all_seperate_test)

    # Compute the generation that a given trajectory terminated at, for each run
    def terminates_at_elem(data):
        first_success = np.argmax((data >= -200), axis=-1)
        first_success[first_success == 0] = 1000
        return first_success + 1

    # Get the median of terminates_at_elem
    def terminates_at(data):
        return np.median(terminates_at_elem(data))

    # Get the best fitness of a given run
    def best_fitness_elem(data):
        return np.max(data, axis=-1)

    # Get the median best fitness of best_fitness_elem
    def best_fitness(data):
        np.median(best_fitness_elem(data))

    print("Self train", terminates_at(all_self_train), best_fitness(all_self_train))
    print("Self test", terminates_at(all_self_test) * 10, best_fitness(all_self_test))
    print(
        "Separate train",
        terminates_at(all_seperate_train),
        best_fitness(all_seperate_train),
    )
    print(
        "Separate test",
        terminates_at(all_seperate_test) * 10,
        best_fitness(all_seperate_test),
    )

    import scipy.stats as stats

    # Apply mann-whitney U for the termination iteration of the population fitness
    mann_term = stats.mannwhitneyu(terminates_at_elem(all_self_train), terminates_at_elem(all_seperate_train))
    # Get the U and p statistics from the U test
    u_term = mann_term.statistic
    p_term = mann_term.pvalue
    # Cliff's delta = 2U/mn - 1 where m = n = num experiments
    # Mann-whitney A = (Cliff's delta + 1) / 2
    a_term = u_term / 100
    print("Termination p", p_term)
    print("Termination u", u_term)
    print("Termination A", a_term)
