import numpy as np
import pandas as pd
import scipy.signal as sg
from .datawriter import DataWriter
from ..ccg import correlograms
from pathlib import Path


class Neurons(DataWriter):
    """Class to hold a group of spiketrains and their labels, ids etc."""

    def __init__(
        self,
        spiketrains=None,
        labels=None,
        ids=None,
        shankids=None,
        waveforms=None,
        filename=None,
    ) -> None:
        super().__init__(filename=filename)

        self.spiketrains = spiketrains
        self.shankid = shankids
        self.labels = labels
        self.ids = ids
        self.waveforms = waveforms
        self.instfiring = None
        self.metadata = None

    @property
    def n_neurons(self):
        return len(self.spiketrains)

    def get_spiketrains(self, ids):
        spiketrains = [self.spiketrains[_] for _ in ids]
        return spiketrains

    def get_shankids(self, ids):
        return np.array([self.shankids[_] for _ in ids])

    def time_slice(self):
        pass

    def _check_neurons(self):
        assert isinstance(self.spiketrains, list)
        # n_neurons = self.n_neurons
        # assert all(
        #     len(arr) == n_neurons
        #     for arr in [
        #         self.shankid,
        #         self.labels,
        #         self.ids,
        #         self.waveforms,
        #         self.instfiring,
        #     ]
        # )

    def load(self):
        data = super().load()
        if data is not None:
            for key in data:
                setattr(self, key, data[key])

    def save(self):

        self._check_neurons()

        data = {
            "spiketrains": self.spiketrains,
            "labels": self.labels,
            "ids": self.ids,
            "wavforms": self.waveforms,
            "instfiring": self.instfiring,
            "metadata": self.metadata,
        }
        super().save(data)

    def firing_rate(self, period, ids=None):
        if ids is None:
            ids = self.ids

        spikes = self.get_spiketrains(ids)
        duration = np.diff(period)
        return (
            np.concatenate([np.histogram(_, bins=period)[0] for _ in spikes]) / duration
        )

    def binned_spiketrains(self, ids, period, binsize=0.25):
        """Get binned spike counts within a period for the given cells"""
        bins = np.arange(period[0], period[1] + binsize, binsize)
        return np.asarray(
            [np.histogram(self.spiketrains[_], bins=bins)[0] for _ in ids]
        )

    def _gaussian(self):
        """Gaussian function for generating instantenous firing rate

        Returns:
            [array] -- [gaussian kernel centered at zero and spans from -1 to 1 seconds]
        """

        sigma = 0.020
        binSize = 0.001
        t_gauss = np.arange(-1, 1, binSize)
        A = 1 / np.sqrt(2 * np.pi * sigma ** 2)
        gaussian = A * np.exp(-(t_gauss ** 2) / (2 * sigma ** 2))

        return gaussian

    def acg(self, spikes=None, bin_size=0.001, window_size=0.05) -> np.ndarray:
        """Get autocorrelogram

        Parameters
        ----------
        spikes : [type], optional
            [description], by default None
        bin_size : float, optional
            [description], by default 0.001
        window_size : float, optional
            [description], by default 0.05
        """

        if isinstance(spikes, np.ndarray):
            spikes = [spikes]
        nCells = len(spikes)

        correlo = []
        for cell in spikes:
            cell_id = np.zeros(len(cell)).astype(int)
            acg = correlograms(
                cell,
                cell_id,
                sample_rate=self._obj.sampfreq,
                bin_size=bin_size,
                window_size=window_size,
            ).squeeze()

            if acg.size == 0:
                acg = np.zeros(acg.shape[-1])

            correlo.append(acg)

        return np.array(correlo)

    def calculate_instfiring(self):
        spkall = np.concatenate(self.spiketrains)
        bins = np.arange(spkall.min(), spkall.max(), 0.001)
        spkcnt = np.histogram(spkall, bins=bins)[0]
        gaussKernel = self._gaussian()
        instfiring = sg.convolve(spkcnt, gaussKernel, mode="same", method="direct")
        self.instfiring = pd.DataFrame({"time": bins[1:], "frate": instfiring})

        self.save()

    def corr_across_time_window(self, cell_ids, period, window=300, binsize=0.25):
        """Correlation of pairwise correlation across a period by dividing into window size epochs

        Parameters
        ----------
        period: array like
            time period where the pairwise correlations are calculated, in seconds
        window : int, optional
            dividing the period into this size window, by default 900
        binsize : float, optional
            [description], by default 0.25
        """

        spikes = self.get_spiketrains(cell_ids)
        epochs = np.arange(period[0], period[1], window)

        pair_corr_epoch = []
        for i in range(len(epochs) - 1):
            epoch_bins = np.arange(epochs[i], epochs[i + 1], binsize)
            spkcnt = np.asarray([np.histogram(x, bins=epoch_bins)[0] for x in spikes])
            epoch_corr = np.corrcoef(spkcnt)
            pair_corr_epoch.append(epoch_corr[np.tril_indices_from(epoch_corr, k=-1)])
        pair_corr_epoch = np.asarray(pair_corr_epoch)

        # masking nan values in the array
        pair_corr_epoch = np.ma.array(pair_corr_epoch, mask=np.isnan(pair_corr_epoch))
        corr = np.ma.corrcoef(pair_corr_epoch)  # correlation across windows
        time = epochs[:-1] + window / 2
        self.corr, self.time = corr, time
        return np.asarray(corr), time

    def corr_pairwise(
        self,
        ids,
        period,
        cross_shanks=False,
        binsize=0.25,
        window=None,
        slideby=None,
    ):
        """Calculates pairwise correlation between given spikes within given period

        Parameters
        ----------
        spikes : list
            list of spike times
        period : list
            time period within which it is calculated , in seconds
        cross_shanks: bool,
            whether restrict pairs to only across shanks, by default False (all pairs)
        binsize : float, optional
            binning of the time period, by default 0.25 seconds

        Returns
        -------
        N-pairs
            pairwise correlations
        """

        spikes = self.get_spiketrains(ids)
        bins = np.arange(period[0], period[1], binsize)
        spk_cnts = np.asarray([np.histogram(cell, bins=bins)[0] for cell in spikes])

        time_points = None
        if window is not None:
            nbins_window = int(window / binsize)
            if slideby is None:
                slideby = window
            nbins_slide = int(slideby / binsize)
            spk_cnts = np.lib.stride_tricks.sliding_window_view(
                spk_cnts, nbins_window, axis=1
            )[:, ::nbins_slide, :]
            time_points = np.lib.stride_tricks.sliding_window_view(
                bins, nbins_window, axis=-1
            )[::nbins_slide, :].mean(axis=-1)

        # ----- indices for cross shanks correlation -------
        shnkId = self.get_shankID(ids)
        selected_pairs = np.tril_indices(len(ids), k=-1)
        if cross_shanks:
            selected_pairs = np.nonzero(
                np.tril(shnkId.reshape(-1, 1) - shnkId.reshape(1, -1))
            )

        corr = []
        if spk_cnts.ndim == 2:
            corr = np.corrcoef(spk_cnts)[selected_pairs]
        else:
            for w in range(spk_cnts.shape[1]):
                corr.append(np.corrcoef(spk_cnts[:, w, :])[selected_pairs])
            corr = np.asarray(corr).T

        return corr, time_points

    def to_neuroscope(self, ids, srate, filename):
        """To view spikes in neuroscope, spikes are exported to .clu.1 and .res.1 files in the basepath.
        You can order the spikes in a way to view sequential activity in neuroscope.

        Parameters
        ----------
        spks : list
            list of spike times.
        """
        spks = self.get_spiketrains(ids)
        nclu = len(spks)
        spk_frame = np.concatenate([(cell * srate).astype(int) for cell in spks])
        clu_id = np.concatenate([[_] * len(spks[_]) for _ in range(nclu)])

        sort_ind = np.argsort(spk_frame)
        spk_frame = spk_frame[sort_ind]
        clu_id = clu_id[sort_ind]
        clu_id = np.append(nclu, clu_id)

        filename = Path(filename)
        file_clu = filename.with_suffix(".clu.1")
        file_res = filename.with_suffix(".res.1")

        with file_clu.open("w") as f_clu, file_res.open("w") as f_res:
            for item in clu_id:
                f_clu.write(f"{item}\n")
            for frame in spk_frame:
                f_res.write(f"{frame}\n")

    def ccg_temporal(self, ids):
        spikes = self.get_spiketrains(ids)
        ccgs = np.zeros((len(spikes), len(spikes), 251)) * np.nan
        spike_ind = np.asarray([_ for _ in range(len(spikes)) if spikes[_].size != 0])
        clus_id = np.concatenate(
            [[_] * len(spikes[_]) for _ in range(len(spikes))]
        ).astype(int)
        sort_ind = np.argsort(np.concatenate(spikes))
        spikes = np.concatenate(spikes)[sort_ind]
        clus_id = clus_id[sort_ind]
        ccgs_ = correlograms(
            spikes,
            clus_id,
            sample_rate=self._obj.sampfreq,
            bin_size=0.001,
            window_size=0.25,
        )
        grid = np.ix_(spike_ind, spike_ind, np.arange(251))
        ccgs[grid] = ccgs_

        center = int(ccgs.shape[-1] / 2) - 1
        diff = ccgs[:, :, center + 2 :].sum(axis=-1) - ccgs[:, :, : center - 2].sum(
            axis=-1
        )
        non_redundant_indx = np.tril_indices_from(diff, k=-1)

        return diff[non_redundant_indx]

    def add_jitter(self):
        pass