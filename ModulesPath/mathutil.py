import numpy as np
import math
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from scipy import stats


def partialcorr(x, y, z):
    """
    correlation between x and y , with controlling for z
    """
    # convert them to pandas series
    x = pd.Series(x)
    y = pd.Series(y)
    z = pd.Series(z)
    # xyz = pd.DataFrame({"x-values": x, "y-values": y, "z-values": z})

    xy = x.corr(y)
    xz = x.corr(z)
    zy = z.corr(y)

    parcorr = (xy - xz * zy) / (np.sqrt(1 - xz ** 2) * np.sqrt(1 - zy ** 2))

    return parcorr


def parcorr_mult(x, y, z):
    """
    correlation between multidimensional x and y , with controlling for multidimensional z

    """

    parcorr = np.zeros((len(z), len(y), len(x)))
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            for k, z_ in enumerate(z):
                parcorr[k, j, i] = partialcorr(x_, y_, z_)

    revcorr = np.zeros((len(z), len(y), len(x)))
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            for k, z_ in enumerate(z):
                revcorr[k, j, i] = partialcorr(x_, z_, y_)

    return parcorr, revcorr


# TODO improve the partial correlation calucalation maybe use arrays instead of list
def parcorr_muglt(x, y, z):
    """
    correlation between multidimensional x and y , with controlling for multidimensional z

    """

    parcorr = np.zeros(z.shape[0], y.shape[0], x.shape[0])
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            for k, z_ in enumerate(z):
                parcorr[k, j, i] = partialcorr(x_, y_, z_)

    revcorr = np.zeros((len(z), len(y), len(x)))
    for i, x_ in enumerate(x):
        for j, y_ in enumerate(y):
            for k, z_ in enumerate(z):
                revcorr[k, j, i] = partialcorr(x_, z_, y_)

    return parcorr, revcorr


def getICA_Assembly(x):
    """extracting statisticaly independent components from significant eigenvectors as detected using Marcenko-Pasteur distributionvinput = Matrix  (m x n) where 'm' are the number of cells and 'n' time bins ICA weights thus extracted have highiest weight positive (as done in Gido M. van de Ven et al. 2016) V = ICA weights for each neuron in the coactivation (weight having the highiest value is kept positive) M1 =  originally extracted neuron weights

    Arguments:
        x {[ndarray]} -- [an array of size n * m]

    Returns:
        [type] -- [Independent assemblies]
    """

    zsc_x = stats.zscore(x, axis=1)

    corrmat = (zsc_x @ zsc_x.T) / x.shape[1]

    lambda_max = (1 + np.sqrt(1 / (x.shape[1] / x.shape[0]))) ** 2
    eig_val, eig_mat = np.linalg.eigh(corrmat)
    get_sigeigval = np.where(eig_val > lambda_max)[0]
    n_sigComp = len(get_sigeigval)
    pca_fit = PCA(n_components=n_sigComp, whiten=False).fit_transform(x)

    ica_decomp = FastICA(n_components=None, whiten=False).fit(pca_fit)
    W = ica_decomp.components_
    V = eig_mat[:, get_sigeigval] @ W.T

    return V


def threshPeriods(sig, lowthresh=1, highthresh=2, minDistance=30, minDuration=50):

    ThreshSignal = np.diff(np.where(sig > lowthresh, 1, 0))
    start = np.where(ThreshSignal == 1)[0]
    stop = np.where(ThreshSignal == -1)[0]

    if start[0] > stop[0]:
        stop = stop[1:]
    if start[-1] > stop[-1]:
        start = start[:-1]

    firstPass = np.vstack((start, stop)).T

    # ===== merging close events
    secondPass = []
    event = firstPass[0]
    for i in range(1, len(firstPass)):
        if firstPass[i, 0] - event[1] < minDistance:
            # Merging states
            event = [event[0], firstPass[i, 1]]
        else:
            secondPass.append(event)
            event = firstPass[i]

    secondPass.append(event)
    secondPass = np.asarray(secondPass)
    event_duration = np.diff(secondPass, axis=1).squeeze()

    # delete very short events
    shortevents = np.where(event_duration < minDuration)[0]
    thirdPass = np.delete(secondPass, shortevents, 0)
    event_duration = np.delete(event_duration, shortevents)

    # keep only events with peak above highthresh
    fourthPass = []
    # peakNormalizedPower, peaktime = [], []
    for i in range(len(thirdPass)):
        maxValue = max(sig[thirdPass[i, 0] : thirdPass[i, 1]])
        if maxValue >= highthresh:
            fourthPass.append(thirdPass[i])
            # peakNormalizedPower.append(maxValue)
            # peaktime.append(
            #     [
            #         secondPass[i, 0]
            #         + np.argmax(zscsignal[secondPass[i, 0] : secondPass[i, 1]])
            #     ]
            # )

    return np.asarray(fourthPass)
