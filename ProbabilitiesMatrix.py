import pandas as pd
import numpy as np
import random

class ProbabilitiesMatrix():

    def __init__(self, data):
        self.data= data
        self.frog_sequence = data['frog_id_kmm']
        self.unique_frogs = sorted(self.frog_sequence.unique())
        self.n_frogs = len(self.unique_frogs)

    def compute_transition_matrix(self, sequence):
        """Compute the transition probability matrix for a given sequence."""
        current_frogs = sequence[:-1]
        next_frogs = sequence.shift(-1)[:-1]
        transition_pairs = pd.DataFrame({'current': current_frogs, 'next': next_frogs})

        # Calculate counts
        transition_counts = pd.crosstab(transition_pairs['current'], transition_pairs['next'])

        # Reindex to include all frogs
        transition_counts = transition_counts.reindex(index=self.unique_frogs, columns=self.unique_frogs, fill_value=0)

        # Normalize to probabilities
        transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)


        return transition_probs

    # Display the transition probability matrix
# import ace_tools as tools; tools.display_dataframe_to_user("Transition Probability Matrix", transition_probabilities)

    def generate_shuffled_sequences(self, n_shuffles):
        """
        Generate shuffled sequences and compute mean transition probabilities.

        Args:
            n_shuffles (int): Number of shuffled sequences to generate.

        Returns:
            mean_matrix (np.ndarray): Mean transition probability matrix.
            std_matrix (np.ndarray): Standard deviation of transition probabilities (optional).
        """
        shuffled_matrices = []
#check rows and columns
        for i in range(n_shuffles):
            shuffled_sequence = self.frog_sequence.sample(frac=1).reset_index(drop=True)  # Shuffle
            shuffled_probs = self.compute_transition_matrix(shuffled_sequence).to_numpy()
            # print("transition_probs for" ,i,":", shuffled_probs)
            shuffled_matrices.append(shuffled_probs)


        # Convert to NumPy array
        # print("all shuffled_metrix before conversion", shuffled_matrices)
        shuffled_matrices = np.array(shuffled_matrices)
        print("******************")
        print(" shuffled_metrix [0] after conversion", shuffled_matrices[0,1,2])
        # Compute mean and standard deviation
        mean_matrix = shuffled_matrices.mean(axis=0)
        std_matrix = shuffled_matrices.std(axis=0)
        # print("mean-matrix", mean_matrix)
        return mean_matrix, std_matrix

    def generate_data_probabilities_matrix(self):
        transition_probabilities= self.compute_transition_matrix(self.frog_sequence)
        return transition_probabilities

csv_path='/Users/Roni_Ariel/Documents/cluster_for_matrix.csv'
data= pd.read_csv(csv_path)

matrix= ProbabilitiesMatrix(data)
data_probabs= matrix.generate_data_probabilities_matrix()

mean_matrix, std_matrix = matrix.generate_shuffled_sequences(n_shuffles=10)
