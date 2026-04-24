import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

"""This script generates example population vectors
1: beginning of maze
2: adjacent bin to beginning of maze
3: middle of maze
4: end of maze but with some activation of beginning of maze place fields"""

pv1 = np.zeros(30)
pv1[0:2] = 6
pv1[2:4] = 8
pv1[4:6] = 4
pv1 + np.random.randn(50) * 0.3
pv1 + np.random.randn(30) * 0.3
pv1 = np.abs(pv1 + np.random.randn(30) * 0.5)
pv2 = np.zeros(30)
pv2[2:4] = 8
pv2[4:6] = 6
pv2[6:8] = 3
pv2 = np.abs(pv2 + np.random.randn(30)*0.5)
pv3 = np.zeros(30)
pv3[-10:-8] = 4
pv3[-12:-10] = 6
pv3[-14:-12] = 4
pv3 = np.abs(pv3 + np.random.randn(30)*0.4)
pv4 = deepcopy(pv1)
pv4[0:4] = pv4[0:4] - 3.5
pv4[4:6] = pv4[4:6] - 1.5
pv4[-2:] = 4
pv4[-4:-2] = 6
pv4[-6:-4] = 4
pv4 = np.abs(pv4 + np.random.randn(30)*0.2)


fig, ax = plt.subplots(1, 4, sharex=True)
for a, pv in zip(ax, [pv1, pv2, pv3, pv4]):
    a.barh(y=np.arange(30), width=pv, height=0.9)
    a.set_xlabel("FR (Hz)")
    a.set_ylabel("Neuron #")
sns.despine(fig=fig)
plt.show()