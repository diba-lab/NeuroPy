import numpy as np
from neuropy.core import Position, Neurons
from neuropy.analyses import Pf1D


def test_pf():
    t = np.linspace(0, 1, 240000)
    y = np.sin(2 * np.pi * 12 * t) * 100

    pos = Position(traces=y.reshape(1, -1), t_start=0, sampling_rate=120)

    spktrns = []
    for i in range(-100, 100, 30):
        indices = np.where((pos.x >= i) & (pos.x <= i + 20))[0]
        indices = np.random.choice(indices, 4000)
        spktrns.append(indices / 120)
    spktrns = np.array(spktrns)

    neurons = Neurons(spiketrains=spktrns, t_start=0, t_stop=2000)
    pf1d = Pf1D(neurons=neurons, position=pos, speed_thresh=0.1, grid_bin=5)
