import numpy as np
from .datawriter import DataWriter


class MultiArmedBandit(DataWriter):
    """
    A class to hold a multi-armed bandit task data.

    """

    def __init__(self, n_ports, probs, metadata=None):

        super().__init__(metadata=metadata)

        self.n_ports = n_ports
        if probs is None:
            self.probs = np.random.rand(n_ports)
        else:
            if len(probs) != n_ports:
                raise ValueError("Length of probs must match the number of arms.")
            self.probs = probs

    def get_probs(self):
        """
        Get the probs of reward for each arm.

        Returns:
            list: Probabilities of reward for each arm.
        """
        return self.probs

    def from_csv(self):
        return None
