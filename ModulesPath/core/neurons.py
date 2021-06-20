import numpy as np
import pandas as pd
import scipy.signal as sg
from .datawriter import DataWriter
from ..utils.ccg import correlograms
from pathlib import Path


class Neurons(DataWriter):
    """Class to hold a group of spiketrains and their labels, ids etc."""

    def __init__(
        self,
        spiketrains,
        t_stop,
        t_start=0.0,
        sampling_rate=1,
        neuron_type=None,
        shankids=None,
        waveforms=None,
        peak_channel=None,
        filename=None,
    ) -> None:
        super().__init__(filename=filename)

        self.spiketrains = spiketrains
        self.shankids = shankids
        self.ids = np.arange(len(self.spiketrains))
        self.waveforms = waveforms
        self.instfiring = None
        self._sampling_rate = sampling_rate
        self.t_start = t_start
        self.t_stop = t_stop
        self.metadata = {}

    @property
    def sampling_rate(self):
        return self._sampling_rate

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

    def to_dict(self):

        self._check_neurons()

        data = {
            "spiketrains": self.spiketrains,
            "labels": self.labels,
            "ids": self.ids,
            "shankids": self.shankids,
            "waveforms": self.waveforms,
            "instfiring": self.instfiring,
            "sampling_rate": self.sampling_rate,
            "metadata": self.metadata,
        }
        return data

    @staticmethod
    def from_dict(d):

        spiketrains = d["spiketrains"]
        neuron_type = d["neuron_type"]

        neurons = Neurons(spiketrains)
        return Neurons

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

    def estimate_neuron_type(self):
        """Auto label cell type using firing rate, burstiness and waveform shape followed by kmeans clustering.

        Reference
        ---------
        Csicsvari, J., Hirase, H., Czurko, A., & Buzsáki, G. (1998). Reliability and state dependence of pyramidal cell–interneuron synapses in the hippocampus: an ensemble approach in the behaving rat. Neuron, 21(1), 179-189.
        """
        spikes = self.times
        self.info["celltype"] = None
        ccgs = self.get_acg(spikes=spikes, bin_size=0.001, window_size=0.05)
        ccg_width = ccgs.shape[-1]
        ccg_center_ind = int(ccg_width / 2)

        # -- calculate burstiness (mean duration of right ccg)------
        ccg_right = ccgs[:, ccg_center_ind + 1 :]
        t_ccg_right = np.arange(ccg_right.shape[1])  # timepoints
        mean_isi = np.sum(ccg_right * t_ccg_right, axis=1) / np.sum(ccg_right, axis=1)

        # --- calculate frate ------------
        frate = np.asarray([len(cell) / np.ptp(cell) for cell in spikes])

        # ------ calculate peak ratio of waveform ----------
        templates = self.templates
        waveform = np.asarray(
            [cell[np.argmax(np.ptp(cell, axis=1)), :] for cell in templates]
        )
        n_t = waveform.shape[1]  # waveform width
        center = np.int(n_t / 2)
        wave_window = int(0.25 * (self._obj.sampfreq / 1000))
        from_peak = int(0.18 * (self._obj.sampfreq / 1000))
        left_peak = np.trapz(
            waveform[:, center - from_peak - wave_window : center - from_peak], axis=1
        )
        right_peak = np.trapz(
            waveform[:, center + from_peak : center + from_peak + wave_window], axis=1
        )

        diff_auc = left_peak - right_peak

        # ---- refractory contamination ----------
        isi = [np.diff(_) for _ in spikes]
        isi_bin = np.arange(0, 0.1, 0.001)
        isi_hist = np.asarray([np.histogram(_, bins=isi_bin)[0] for _ in isi])
        n_spikes_ref = np.sum(isi_hist[:, :2], axis=1) + 1e-16
        ref_period_ratio = (np.max(isi_hist, axis=1) / n_spikes_ref) * 100
        mua_cells = np.where(ref_period_ratio < 400)[0]
        good_cells = np.where(ref_period_ratio >= 400)[0]

        self.info.loc[mua_cells, "celltype"] = "mua"

        param1 = frate[good_cells]
        param2 = mean_isi[good_cells]
        param3 = diff_auc[good_cells]

        features = np.vstack((param1, param2, param3)).T
        features = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=2).fit(features)
        y_means = kmeans.predict(features)

        interneuron_label = np.argmax(kmeans.cluster_centers_[:, 0])
        intneur_id = np.where(y_means == interneuron_label)[0]
        pyr_id = np.where(y_means != interneuron_label)[0]
        self.info.loc[good_cells[intneur_id], "celltype"] = "intneur"
        self.info.loc[good_cells[pyr_id], "celltype"] = "pyr"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            frate[mua_cells],
            mean_isi[mua_cells],
            diff_auc[mua_cells],
            c=self.colors["mua"],
            s=50,
            label="mua",
        )

        ax.scatter(
            param1[pyr_id],
            param2[pyr_id],
            param3[pyr_id],
            c=self.colors["pyr"],
            s=50,
            label="pyr",
        )

        ax.scatter(
            param1[intneur_id],
            param2[intneur_id],
            param3[intneur_id],
            c=self.colors["intneur"],
            s=50,
            label="int",
        )
        ax.legend()
        ax.set_xlabel("Firing rate (Hz)")
        ax.set_ylabel("Mean isi (ms)")
        ax.set_zlabel("Difference of \narea under shoulders")

        data = np.load(self.files.spikes, allow_pickle=True).item()
        data["info"] = self.info
