import numpy as np


def bayesian_decoder(self, spkcount, ratemaps):
    """
    ===========================
    Probability is calculated using this formula
    prob = (1 / nspike!)* ((tau * frate)^nspike) * exp(-tau * frate)
    where,
        tau = binsize
    ===========================
    """
    tau = self.binsize
    nCells = spkcount.shape[0]
    cell_prob = np.zeros((ratemaps.shape[1], spkcount.shape[1], nCells))
    for cell in range(nCells):
        cell_spkcnt = spkcount[cell, :][np.newaxis, :]
        cell_ratemap = ratemaps[cell, :][:, np.newaxis]

        coeff = 1 / (factorial(cell_spkcnt))
        # broadcasting
        cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (
            np.exp(-tau * cell_ratemap)
        )

    posterior = np.prod(cell_prob, axis=2)
    posterior /= np.sum(posterior, axis=0)

    return posterior
