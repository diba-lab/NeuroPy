import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.gridspec as gridspec
import matplotlib as mpl
from ccg import correlograms


class event_event:
    def compute(
        self,
        ref,
        event,
        quantparam=None,
        binsize=0.01,
        window=1,
        fs=1250,
        nQuantiles=10,
        period=None,
    ):
        """psth of 'event' with respect to 'ref'

        Args:
            ref (array): 1-D array of timings of reference event in seconds
            event (1D array): timings of events whose psth will be calculated
            quantparam (1D array): values used to divide 'ref' into quantiles
            binsize (float, optional): [description]. Defaults to 0.01.
            window (int, optional): [description]. Defaults to 1.
            nQuantiles (int, optional): [description]. Defaults to 10.

        Returns:
            [type]: [description]
        """

        # --- parameters----------
        if period is not None:
            event = event[(event > period[0]) & (event < period[1])]
            if quantparam is not None:
                quantparam = quantparam[(ref > period[0]) & (ref < period[1])]
            ref = ref[(ref > period[0]) & (ref < period[1])]

        if quantparam is not None:
            assert len(event) == len(quantparam), print("length must be same")
            quantiles = pd.qcut(quantparam, nQuantiles, labels=False)

            quants, eventid = [], []
            for category in range(nQuantiles):
                indx = np.where(quantiles == category)[0]
                quants.append(ref[indx])
                eventid.append(category * np.ones(len(indx)).astype(int))

            quants.append(event)
            eventid.append(((nQuantiles + 1) * np.ones(len(event))).astype(int))

            quants = np.concatenate(quants)
            eventid = np.concatenate(eventid)
        else:
            quants = np.concatenate((ref, event))
            eventid = np.concatenate(
                [np.ones(len(ref)), 2 * np.ones(len(event))]
            ).astype(int)

        sort_ind = np.argsort(quants)

        ccg = correlograms(
            quants[sort_ind],
            eventid[sort_ind],
            sample_rate=fs,
            bin_size=binsize,
            window_size=window,
        )

        self.psth = ccg[:-1, -1, :]

        return self.psth

    def plot(self, ax=None):

        if self.psth.ndim == 1:
            psth = self.psth[np.newaxis, :]
        else:
            psth = self.psth

        nQuantiles = self.psth.shape[0]
        cmap = mpl.cm.get_cmap("viridis")
        colmap = [cmap(x) for x in np.linspace(0, 1, nQuantiles)]

        if ax is None:
            plt.clf()
            fig = plt.figure(num=None, figsize=(10, 15))
            gs = gridspec.GridSpec(1, 1, figure=fig)
            fig.subplots_adjust(hspace=0.3)
            ax = fig.add_subplot(gs[0])

        for quant in range(nQuantiles):
            ax.plot(psth[quant, :], color=colmap[quant])
        ax.set_xlabel("Time from hswa (s)")
        ax.set_ylabel("Counts")
        # ax.set_title(self._obj.sessinfo.session.sessionName)

    def plot_raster(self):
        pass

