import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed
from neuropy.core.epoch import Epoch
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import PCA
import scipy.signal as sg

from .. import core
from ..plotting import plot_epochs
from ..utils import mathutil, signal_process


def hmmfit1d(Data, ret_means=False, **kwargs):
    # hmm states on 1d data and returns labels with highest mean = highest label
    flag = None
    if np.isnan(Data).any():
        nan_indices = np.where(np.isnan(Data) == 1)[0]
        non_nan_indices = np.where(np.isnan(Data) == 0)[0]
        Data_og = Data
        Data = np.delete(Data, nan_indices)
        hmmlabels = np.nan * np.ones(len(Data_og))
        flag = 1

    Data = (np.asarray(Data)).reshape(-1, 1)
    models = []
    scores = []
    for i in range(10):
        model = GaussianHMM(n_components=2, n_iter=10, random_state=i, **kwargs)
        model.fit(Data)
        models.append(model)
        scores.append(model.score(Data))
    model = models[np.argmax(scores)]

    hidden_states = model.predict(Data)
    mus = np.squeeze(model.means_)
    sigmas = np.squeeze(np.sqrt(model.covars_))
    transmat = np.array(model.transmat_)

    idx = np.argsort(mus)
    mus = mus[idx]
    sigmas = sigmas[idx]
    transmat = transmat[idx, :][:, idx]

    state_dict = {}
    states = [i for i in range(4)]
    for i in idx:
        state_dict[idx[i]] = states[i]

    relabeled_states = np.asarray([state_dict[h] for h in hidden_states])
    relabeled_states[:2] = [0, 0]
    relabeled_states[-2:] = [0, 0]

    if flag:
        hmmlabels[non_nan_indices] = relabeled_states

    else:
        hmmlabels = relabeled_states

    if ret_means:
        return hmmlabels, mus
    else:
        return hmmlabels


def gaussian_classify(feat, ret_params=False, plot=False, ax=None):
    if feat.ndim < 2:
        feat = feat[:, None]
    clus = GaussianMixture(
        n_components=2, init_params="k-means++", max_iter=200, n_init=10
    ).fit(feat)
    labels = clus.predict(feat)
    clus_means = clus.means_[:, 0]

    # --- order cluster labels by increasing mean (low=0, high=1) ------
    sort_idx = np.argsort(clus_means)
    label_map = np.zeros_like(sort_idx)
    label_map[sort_idx] = np.arange(len(sort_idx))
    fixed_labels = label_map[labels.astype("int")]

    if plot:
        if feat.ndim == 2:
            label_bool = fixed_labels.astype("bool")
            ax.scatter(feat[label_bool, 0], feat[label_bool, 1], s=1)
            ax.scatter(feat[~label_bool, 0], feat[~label_bool, 1], s=1)

    if ret_params:
        params_dict = dict(
            weights=clus.weights_[sort_idx],
            means=clus.means_[sort_idx],
            covariances=clus.covariances_[sort_idx, :, :],
        )
        return fixed_labels, params_dict
    else:
        return fixed_labels


def correlation_emg(
    signal: core.Signal,
    window,
    overlap,
    probe: core.ProbeGroup = None,
    min_dist=0,
    n_jobs=8,
):
    """Calculating emg

    Parameters
    ----------
    window : int
        window size in seconds
    overlap : float
        overlap between windows in seconds
    n_jobs: int,
        number of cpu/processes to use

    Returns
    -------
    array
        emg calculated at each time window
    """
    print("starting emg calculation")
    highfreq = 600
    lowfreq = 300
    t_start = signal.t_start
    t_stop = signal.t_stop
    sRate = signal.sampling_rate
    emg_chans = signal.channel_id

    if probe is None:
        pairs_bool = np.ones((len(emg_chans), len(emg_chans))).astype("bool")
    elif isinstance(probe, core.ProbeGroup):
        # changrp = np.concatenate(probe.get_connected_channels(groupby="probe"))
        probe_df = probe.to_dataframe()
        probe_df = probe_df[probe_df.connected == True]
        probe_df_chans = list(probe_df["channel_id"].values)
        x, y = probe_df.x.values.astype("float"), probe_df.y.values.astype("float")
        # --- choosing pairs of channels spaced min_dist --------
        squared_diff = lambda arr: (arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2
        distance = np.sqrt(squared_diff(x) + squared_diff(y))

        emg_chans = emg_chans[np.isin(emg_chans, probe_df_chans)]
        chan_probe_indx = [probe_df_chans.index(chan) for chan in emg_chans]
        emg_chans_distance = distance[np.ix_(chan_probe_indx, chan_probe_indx)]
        pairs_bool = emg_chans_distance > min_dist
    else:
        raise ValueError("invalid probe input")

    timepoints = np.arange(t_start, t_stop - window, window - overlap)

    # ---- Mean correlation across all selected channels calculated in parallel --
    def corrchan(start):
        lfp_req = np.asarray(
            signal.time_slice(
                channel_id=emg_chans, t_start=start, t_stop=start + window
            ).traces
        )
        yf = signal_process.filter_sig.bandpass(
            lfp_req, lf=lowfreq, hf=highfreq, fs=sRate
        )
        # ltriang = np.tril_indices(n_emg_chans, k=-1)
        ltriang = np.tril(pairs_bool, k=-1)
        return np.corrcoef(yf)[ltriang].mean()

    corr_per_window = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(corrchan)(start) for start in timepoints
    )
    emg_lfp = np.asarray(corr_per_window)

    print("emg calculation done")
    return emg_lfp


def get_artifact_indices(arr, thresh=3):
    zsc_arr = stats.zscore(arr)
    # if thresh is None:
    #     thresh = stats.scoreatpercentile()
    zsc_arr = np.where(zsc_arr > 0, zsc_arr, 0)
    _, peaks_prop = sg.find_peaks(
        zsc_arr,
        height=thresh,
        prominence=0,
        plateau_size=1,
    )
    left_base = peaks_prop["left_bases"]
    right_base = peaks_prop["right_bases"]

    return np.unique(
        np.concatenate([np.arange(a, b + 1) for a, b in zip(left_base, right_base)])
    )


def interpolate_indices(arr, indices):
    new_arr = arr.copy()
    new_arr[indices] = np.nan
    return pd.DataFrame(new_arr).interpolate(method="linear")[0].to_numpy()


def detect_brainstates_epochs(
    signal: core.Signal,
    probe: core.ProbeGroup,
    window=1,
    overlap=0.2,
    sigma=3,
    emg_signal=None,
    theta_channel=None,
    delta_channel=None,
    min_dur=6,
    ignore_epochs: core.Epoch = None,
    plot=False,
    fp_bokeh_plot=None,
):
    """detects sleep states using LFP.

    Parameters
    ----------
    signal : core.Signal object
        Signal object containing LFP traces across all good channels preferrable
    probe: core.Probe object
        probemap containing channel coordinates and ids of good and bad channels
    window : float, optional
        window size for spectrogram calculation, by default 1 seconds
    overlap : float, optional
        by how much the adjacent windows overlap, by default 0.2 seconds
    emg_signal : bool, optional
        if emg has already been calculated pass that as a signal object, by None
    theta_channel: bool, optional
        specify channel_id that can be used for theta power calculation, by None
    delta_channel: bool, optional
        specify channel that can be used for slow wave calculation, by None
    min_dur: bool, optional
        minimum duration of each state, by None
    ignore_epochs: bool, optional
        epochs which are ignored during scoring, could be noise epochs, by None
    fp_bokeh_plot: Path to file, optional
        if given then .html file is saved detailing some of scoring parameters and classification, by None
    """

    freqs = np.geomspace(1, 100, 100)
    spect_kw = dict(window=window, overlap=overlap, freqs=freqs, norm_sig=True)

    smooth_ = lambda arr: gaussian_filter1d(arr, sigma=sigma / (window - overlap))
    print(f"channel for sleep detection: {theta_channel,delta_channel}")

    # ---- theta-delta ratio calculation -----
    if theta_channel is not None:
        theta_signal = signal.time_slice(channel_id=[theta_channel])
        theta_chan_sg = signal_process.FourierSg(theta_signal, **spect_kw)
        theta_chan_zscored_spect = stats.zscore(np.log10(theta_chan_sg.traces), axis=1)

    if delta_channel is not None:
        if delta_channel == theta_channel:
            delta_chan_sg = theta_chan_sg
        else:
            delta_signal = signal.time_slice(channel_id=[delta_channel])
            delta_chan_sg = signal_process.FourierSg(delta_signal, **spect_kw)

        delta_chan_zscored_spect = stats.zscore(np.log10(delta_chan_sg.traces), axis=1)

    time = theta_chan_sg.time

    print(f"spectral properties calculated")

    # ---- emg processing ----
    if emg_signal is None:
        # emg_kw = dict(window=window, overlap=overlap)
        emg_kw = dict(window=1, overlap=0)
        emg = correlation_emg(signal=signal, probe=probe, **emg_kw)
        emg = gaussian_filter1d(emg, sigma=10)
        emg_t = np.linspace(signal.t_start, signal.t_stop, len(emg))
        emg = np.interp(theta_chan_sg.time, emg_t, emg)
    elif isinstance(emg_signal, core.Signal):
        assert emg_signal.n_channels == 1, "emg_signal should only have one channel"
        emg_trace = emg_signal.traces[0]
        emg_srate = emg_signal.sampling_rate
        # Smoothing emg with 10 seconds gaussian kernel, works better
        emg_trace = gaussian_filter1d(emg_trace, sigma=10 * emg_srate)
        emg = np.interp(time, emg_signal.time, emg_trace)
    else:
        print("Using emg_channel has not been implemented yet")
        # TODO: if one of the channels provides emg, use that
        # emg = signal.time_slice(emg_channel)

    # ----- note indices from ignore_epochs and include spectogram noisy timepoints ------
    noisy_bool = np.zeros_like(time).astype("bool")
    if ignore_epochs is not None:
        noisy_arr = ignore_epochs.as_array()
        for noisy_ind in range(noisy_arr.shape[0]):
            st = noisy_arr[noisy_ind, 0]
            en = noisy_arr[noisy_ind, 1]
            noisy_bool[
                np.where((time >= st - window) & (time <= en + window))[0]
            ] = True

    noisy_spect_bool = np.logical_or(
        stats.zscore(np.abs(theta_chan_zscored_spect.sum(axis=0))) >= 5,
        stats.zscore(np.abs(delta_chan_zscored_spect.sum(axis=0))) >= 5,
    )
    noisy_bool = np.logical_or(noisy_bool, noisy_spect_bool)

    # ------ features to be used for scoring ---------
    theta = theta_chan_sg.get_band_power(5, 10)
    theta[~noisy_bool] = smooth_(theta[~noisy_bool])
    bp_2to16 = theta_chan_sg.get_band_power(2, 16)
    bp_2to16[~noisy_bool] = smooth_(bp_2to16[~noisy_bool])

    delta = delta_chan_sg.get_band_power(1, 4)
    delta[~noisy_bool] = smooth_(delta[~noisy_bool])

    theta_delta_ratio = theta / delta  # used for REM vs NREM
    theta_dominance = theta / bp_2to16  # buzsaki lab, used for AW vs QW

    # Usually the first principal component has highiest weights in lower frequency band (1-32 Hz)
    broadband_sw = np.zeros_like(theta)
    broadband_sw[~noisy_bool] = smooth_(
        PCA(n_components=1)
        .fit_transform(delta_chan_zscored_spect[:, ~noisy_bool].T)
        .squeeze()
    )

    # ----transform (if any) parameters such zscoring, minmax scaling etc. ----
    emg = np.log10(emg)
    emg[np.isnan(emg)] = np.nanmax(emg)
    thdom_bsw_arr = np.vstack((theta_dominance, broadband_sw)).T

    # ---- Clustering---------

    emg_bool = np.zeros_like(emg).astype("bool")
    emg_labels, emg_fit_params = gaussian_classify(emg[~noisy_bool], ret_params=True)
    emg_bool[~noisy_bool] = emg_labels.astype("bool")

    nrem_rem_bool = np.zeros_like(emg).astype("bool")
    nrem_rem_labels, nrem_rem_fit_params = gaussian_classify(
        theta_delta_ratio[~noisy_bool & ~emg_bool], ret_params=True
    )
    nrem_rem_bool[~noisy_bool & ~emg_bool] = nrem_rem_labels.astype("bool")

    # Using only theta ratio column to separate active and quiet awake
    aw_qw_bool = np.zeros_like(emg).astype("bool")
    aw_qw_label, aw_qw_fit_params = gaussian_classify(
        theta_dominance[~noisy_bool & emg_bool], ret_params=True
    )
    aw_qw_bool[~noisy_bool & emg_bool] = aw_qw_label.astype("bool")

    # --- states: Active wake (AW), Quiet Wake (QW), REM, NREM
    # initialize empty label states
    states = np.array([""] * len(time), dtype="U4")
    states[~noisy_bool & emg_bool & aw_qw_bool] = "AW"
    states[~noisy_bool & emg_bool & ~aw_qw_bool] = "QW"
    states[~noisy_bool & ~emg_bool & nrem_rem_bool] = "REM"
    states[~noisy_bool & ~emg_bool & ~nrem_rem_bool] = "NREM"

    # --- TODO micro-arousals ---------
    # ma_bool = emg_bool & delta_bool
    # states[ma_indices] = "MA"

    # ---- removing very short epochs -----
    epochs = Epoch.from_string_array(states, t=time)
    metadata = {
        "window": window,
        "overlap": overlap,
        "theta_channel": theta_channel,
        "delta_channel": delta_channel,
    }
    epochs = epochs.duration_slice(min_dur=min_dur)
    epochs = epochs.fill_blank("from_left")  # this will also fill ignore_epochs

    # clearing out ignore_epochs
    if ignore_epochs is not None:
        for e in ignore_epochs.as_array():
            epochs = epochs.delete_in_between(e[0], e[1])

    epochs.metadata = metadata

    if plot:
        _, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True)
        params = [broadband_sw, theta_delta_ratio, emg]
        params_names = ["Delta", "Theta ratio", "EMG"]

        for i, (param, name) in enumerate(zip(params, params_names)):
            axs[i].plot(time, param, "k.", markersize=1)
            # axs[0].set_ylim([-1, 4])
            axs[i].set_title(name, loc="left")

        states_colors = dict(NREM="#e920e2", REM="#f7abf4", QW="#fbc77e", AW="#e28708")
        plot_epochs(
            epochs=epochs,
            ax=axs[3],
            # labels_order=["NREM", "REM", "QW", "AW"],
            # colors=states_colors,
        )

    if fp_bokeh_plot is not None:
        import bokeh.plotting as bplot
        import bokeh.models as bmodels
        from bokeh.layouts import column, row

        bplot.output_file(fp_bokeh_plot, title="Sleep scoring results")
        tools = "pan,box_zoom,reset"
        dimensions = dict(tools=tools, width=1000, height=200)

        def plot_feature(x, y, label):
            feature_kw = dict(
                color="black", line_width=2, alpha=0.7, legend_label=label
            )
            p = bplot.figure(**dimensions, y_axis_label="Arb. units")
            p.line(x[~noisy_bool], y[~noisy_bool], **feature_kw)
            return p

        p_sw = plot_feature(time, broadband_sw, "Broadband slow wave")
        p_theta = plot_feature(time, theta_delta_ratio, "Theta ratio")
        p_theta.x_range = p_sw.x_range
        p_emg = plot_feature(time, emg, "EMG")
        p_emg.x_range = p_sw.x_range

        def plot_thresh(x, fit_params, x_label, low_label, high_label, title):
            p = bplot.figure(
                title=title,
                x_axis_label=x_label,
                y_axis_label="Density",
                width=330,
                height=330,
            )
            bins = np.linspace(x.min(), x.max(), 200)
            hist = np.histogram(x, bins, density=True)[0]
            means = fit_params["means"]
            covs = fit_params["covariances"]
            weights = fit_params["weights"]
            lowfit = (
                stats.norm.pdf(
                    bins[:-1], float(means[0][0]), np.sqrt(float(covs[0][0][0]))
                )
                * weights[0]
            )
            highfit = (
                stats.norm.pdf(
                    bins[:-1], float(means[1][0]), np.sqrt(float(covs[1][0][0]))
                )
                * weights[1]
            )

            p.vbar(
                bins[:-1],
                bottom=0,
                width=np.diff(bins)[0] / 2,
                top=hist,
                color="#e5d02e",
                legend_label="Overall",
            )
            p.line(
                bins[:-1], lowfit, color="gray", legend_label=low_label, line_width=2
            )
            p.line(
                bins[:-1], highfit, color="black", legend_label=high_label, line_width=2
            )
            return p

        p_emg_dist = plot_thresh(
            emg[~noisy_bool],
            fit_params=emg_fit_params,
            x_label="log EMG",
            low_label="Sleep",
            high_label="Wake",
            title="Sleep vs Wake",
        )

        p_nrem_rem_dist = plot_thresh(
            theta_delta_ratio[~noisy_bool & ~emg_bool],
            fit_params=nrem_rem_fit_params,
            x_label="log EMG",
            low_label="NREM",
            high_label="REM",
            title="NREM vs REM (low EMG)",
        )

        p_aw_qw_dist = plot_thresh(
            theta_dominance[~noisy_bool & emg_bool],
            fit_params=aw_qw_fit_params,
            x_label="Theta dominance",
            low_label="Quiet",
            high_label="Active",
            title="Active vs Quiet wake (high EMG)",
        )

        p_states = bplot.figure(x_range=p_sw.x_range, **dimensions)
        p_states.yaxis.visible = False
        y = 0
        colors = ["blue", "tomato", "#64d39c", "#267e52"]
        for i, state in enumerate(["NREM", "REM", "QW", "AW"]):
            state_epochs = epochs[state]
            p_states.rect(
                state_epochs.starts + state_epochs.durations / 2,
                y,
                state_epochs.durations,
                1,
                color=colors[i],
                legend_label=state,
            )
            y += 1

        bplot.save(
            column(
                p_states,
                p_sw,
                p_theta,
                p_emg,
                row(p_emg_dist, p_nrem_rem_dist, p_aw_qw_dist),
            )
        )

    return epochs
