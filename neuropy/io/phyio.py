import numpy as np
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from .. import core


class PhyIO:
    def __init__(self, dirname: Path, include_groups=("mua", "good")) -> None:
        self.source_dir = Path(dirname)
        self.sampling_rate = None
        self.spiketrains = None
        self.waveforms = None
        self.peak_waveforms = None
        self.peak_channels = None
        self.include_groups = include_groups
        self._parse_folder()

    def _parse_folder(self):
        params = {}
        with (self.source_dir / "params.py").open("r") as f:
            for line in f:
                line_values = (
                    line.replace("\n", "")
                    .replace('r"', '"')
                    .replace('"', "")
                    .split("=")
                )
                params[line_values[0].strip()] = line_values[1].strip()

        self.sampling_rate = int(float(params["sample_rate"]))
        self.n_channels = int(params["n_channels_dat"])
        self.n_features_per_channel = int(params["n_features_per_channel"])

        spktime = np.load(self.source_dir / "spike_times.npy")
        clu_ids = np.load(self.source_dir / "spike_clusters.npy")
        spk_templates_id = np.load(self.source_dir / "spike_templates.npy")
        spk_templates = np.load(self.source_dir / "templates.npy")
        cluinfo = pd.read_csv(self.source_dir / "cluster_info.tsv", delimiter="\t")
        similarity = np.load(self.source_dir / "similar_templates.npy")
        waveform_amplitude = np.load(self.source_dir / "amplitudes.npy")
        channel_map = np.load(self.source_dir / "channel_map.npy")
        channel_positions = np.load(self.source_dir / "channel_positions.npy")

        # if self.include_noise_clusters:
        #     cluinfo = cluinfo[
        #         cluinfo["group"].isin(["mua", "good", "noise"])
        #     ].reset_index(drop=True)
        # else:
        #     cluinfo = cluinfo[cluinfo["group"].isin(["mua", "good"])].reset_index(
        #         drop=True
        #     )

        cluinfo = cluinfo[cluinfo["group"].isin(self.include_groups)].reset_index(
            drop=True
        )
        if "id" not in cluinfo:
            print(
                "id column does not exist in cluster_info.tsv. Using cluster_id column instead."
            )
            cluinfo["id"] = cluinfo["cluster_id"]

        self.cluster_info = cluinfo.copy()
        self.neuron_ids = cluinfo["id"]
        self.peak_amplitudes = cluinfo["amp"].values
        self.peak_channels = cluinfo["ch"].values
        self.shank_ids = cluinfo["sh"].values
        self.channel_map = channel_map
        self.channel_positions = channel_positions

        if not self.cluster_info.empty:
            clu_id, spiketrains, template_id = [], [], []
            template_waveforms, template_amplitudes = [], []
            for clu in cluinfo.itertuples():
                clu_spike_location = np.where(clu_ids == clu.id)[0]
                spkframes = spktime[clu_spike_location]
                cell_template_id, counts = np.unique(
                    spk_templates_id[clu_spike_location], return_counts=True
                )
                spiketrains.append(spkframes / self.sampling_rate)
                template_waveforms.append(
                    spk_templates[cell_template_id[np.argmax(counts)]].squeeze().T
                )
                template_amplitudes.append(waveform_amplitude[clu_spike_location])
                clu_id.append(clu_ids[clu_spike_location])
                template_id.append(cell_template_id[np.argmax(counts)])

            self.spiketrains = np.array(spiketrains, dtype="object")
            self.clu_ids = np.array(clu_id, dtype="object")
            self.template_id = np.array(template_id)
            self.waveforms = np.array(template_waveforms)
            self.waveforms_amplitude = np.asarray(template_amplitudes, dtype="object")
            self.peak_waveforms = [
                wav[np.argmax(np.max(wav, axis=1))] for wav in template_waveforms
            ]

    def calculate_metrics(self, epochs: core.Epoch, radius_um=200, max_spikes=1000):
        """Calculating isolation distances and l_ratios for clusters

        Method/definitions/algorithm
        --------
        pc_features.npy : (n_spikes x n_pc_features_per_channel x n_template_channels)
            stores features/projection for each spikes onto its template channels
        pc_feature_ind.npy : (n_templates x n_template_channels)
            number of templates computed by spikesorting algorithm

        After spikesorting is done, the number of templates are fixed. When doing manual curation in phy-gui, the number of clusters are changed because of merging and splitting. However, spikes from merged/splitted cluster still point to their original template. In phy-gui, for merged cluster, template with most number of spikes is shown. In the algorithm below, it loops over each cluster and identifies other clusters which have their peak channels in the neighbourhood defined by radius_um parameter. Now it is possible that neighbouring clusters' template channel may not be exactly the same as the cluster for which metrics are being calculated. For this reason, if other clusters' templates are defined for channels for current cluster, they are considered zeros on those channels.

        Parameters
        ----------
        epochs : core.Epoch
            calculate metrics in each epoch
        radius_um : int, optional
            only units with peak channels within radius in um, by default 200
        max_spikes : int, optional
            maximum number of spikes for each cluster, by default 1000
        """
        epochs = epochs.to_dataframe()
        spike_times = np.concatenate(self.spiketrains)
        spike_clusters = np.concatenate(self.clu_ids)
        pc_features = np.load(self.source_dir / "pc_features.npy", mmap_mode="r")
        n_pc = pc_features.shape[1]
        pc_feature_ind = np.load(self.source_dir / "pc_feature_ind.npy")
        template_id = self.tempate_id
        channel_map = self.channel_map
        channel_positions = self.channel_positions
        peak_channels = self.peak_channels

        # -------- calculating distance between each pair of channels ---------------
        x, y = channel_positions.T
        x_dist = x[:, np.newaxis] - x[np.newaxis, :]
        y_dist = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(x_dist ** 2 + y_dist ** 2)

        # ids of cluster for which to calculate the metrics
        cluster_ids = self.cluster_info.id
        metrics = pd.DataFrame()
        for epoch in epochs.itertuples():

            # initializing metrics
            isolation_distances = np.zeros(len(cluster_ids))
            l_ratios = np.zeros(len(cluster_ids))

            # masking spike_times outside current epoch
            outside_epoch = np.logical_or(
                spike_times < epoch.start, spike_times > epoch.stop
            )
            spike_clusters_epoch = np.ma.array(spike_clusters, mask=outside_epoch)

            for idx, cluster_id in enumerate(cluster_ids):

                # calculate peak channel for this cluster
                peak_chan = peak_channels[idx]

                # identify template channels (non-zero channels)
                template_channels = pc_feature_ind[template_id[idx]]
                template_channels = np.trim_zeros(template_channels, "b")
                n_template_chans = len(template_channels)

                # peak channels within radius_um
                other_chan_distance = dist_matrix[np.where(channel_map == peak_chan)[0]]
                other_chan_distance = other_chan_distance.reshape(-1)
                other_peak_chans = channel_map[other_chan_distance <= radius_um]

                # clusters with peak channel within radius and includes this cluster
                other_cluster_indx = np.where(np.isin(peak_channels, other_peak_chans))[
                    0
                ]
                other_cluster_ids = cluster_ids[other_cluster_indx]

                all_pcs = np.zeros((0, n_pc, n_template_chans))
                all_labels = np.zeros((0,))

                for indx, cluster_id2 in zip(other_cluster_indx, other_cluster_ids):
                    # identify template channels for other cluster
                    template_channels_other = np.trim_zeros(
                        pc_feature_ind[template_id[indx]], "b"
                    )

                    spike_loc = np.ma.where(spike_clusters_epoch == cluster_id2)[0]

                    if len(spike_loc) > max_spikes:
                        spike_loc = np.sort(
                            np.random.permutation(spike_loc)[:max_spikes]
                        )

                    n_spikes = len(spike_loc)

                    # only select principal component projects for overlaping channels, projections are considered zero for non-overlaping channels
                    pcs = np.zeros((n_spikes, n_pc, n_template_chans))
                    _, clu_ind, other_ind = np.intersect1d(
                        template_channels,
                        template_channels_other,
                        assume_unique=True,
                        return_indices=True,
                    )
                    pc_cluster_id2 = pc_features[spike_loc, :, :]
                    pcs[:, :, clu_ind] = pc_cluster_id2[:, :, other_ind]
                    labels = np.ones((n_spikes,)) * cluster_id2

                    all_pcs = np.concatenate((all_pcs, pcs), 0)
                    all_labels = np.concatenate((all_labels, labels), 0)

                all_pcs = np.reshape(
                    all_pcs, (all_pcs.shape[0], pc_features.shape[1] * n_template_chans)
                )

                isolation_distances[idx], l_ratios[idx] = mahalanobis_metrics(
                    all_pcs, all_labels, cluster_id
                )

            metrics = metrics.append(
                pd.DataFrame(
                    {
                        "cluster_id": cluster_ids,
                        "isolation_distances": isolation_distances,
                        "l_ratios": l_ratios,
                        "epoch": epoch.label,
                    }
                )
            )
        self.metrics = metrics


def mahalanobis_metrics(all_pcs, all_labels, this_unit_id):
    """Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)
    Based on metrics described in Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11
    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    Outputs:
    --------
    isolation_distance : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit

    NOTE: This code was obtained from spikeinterfce github repo
    Link: https://github.com/SpikeInterface/spikemetrics
    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id, :]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]

    mean_value = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.linalg.LinAlgError:  # case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(
        cdist(mean_value, pcs_for_other_units, "mahalanobis", VI=VI)[0]
    )

    mahalanobis_self = np.sort(
        cdist(mean_value, pcs_for_this_unit, "mahalanobis", VI=VI)[0]
    )

    n = np.min(
        [pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]]
    )  # number of spikes

    if n >= 2:
        dof = pcs_for_this_unit.shape[1]  # number of features
        l_ratio = (
            np.sum(1 - chi2.cdf(pow(mahalanobis_other, 2), dof))
            / mahalanobis_self.shape[0]
        )
        isolation_distance = pow(mahalanobis_other[n - 1], 2)
        # if math.isnan(l_ratio):
        #     print("NaN detected", mahalanobis_other, VI)
    else:
        l_ratio = np.nan
        isolation_distance = np.nan

    return isolation_distance, l_ratio
