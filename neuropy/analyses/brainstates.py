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
    """detects sleep states for the recording

    Parameters
    ----------
    chans : int, optional
        channel you want to use for sleep detection, by default None
    window : int, optional
        bin size, by default 1
    overlap : float, optional
        seconds of overlap between adjacent window , by default 0.2
    emgfile : bool, optional
        if True load the emg file in the basepath, by default False
    behavior_epochs : None,
        These are eopchs when the animal was put on a track for behavior.
        Using wavelet these epochs can be further fine tuned for accurate active and quiet period.
    """

    freqs = np.geomspace(1, 100)
    spect_kw = dict(window=window, overlap=overlap, freqs=freqs, norm_sig=True)

    smooth_ = lambda arr: gaussian_filter1d(arr, sigma=sigma / (window - overlap))
    print(f"channel for sleep detection: {theta_channel,delta_channel}")

    # ---- theta-delta ratio calculation -----
    if theta_channel is not None:
        theta_signal = signal.time_slice(channel_id=[theta_channel])
        theta_chan_sg = signal_process.FourierSg(theta_signal, **spect_kw)
        theta = smooth_(theta_chan_sg.get_band_power(5, 10))
        theta_ratio = theta / smooth_(theta_chan_sg.get_band_power(2, 16))

    if delta_channel is not None:
        if delta_channel == theta_channel:
            zscored_spect = stats.zscore(theta_chan_sg.traces, axis=1)
        else:
            delta_signal = signal.time_slice(channel_id=[delta_channel])
            delta_chan_sg = signal_process.FourierSg(delta_signal, **spect_kw)
            zscored_spect = stats.zscore(delta_chan_sg.traces, axis=1)

        # Usually the first principal component has highiest weights in lower frequency band (1-32 Hz)
        broadband_sw = smooth_(
            PCA(n_components=1).fit_transform(zscored_spect.T).squeeze()
        )

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

    # ----- note indices from ignore_epochs ------
    noisy_bool = np.zeros_like(theta_ratio).astype("bool")
    if ignore_epochs is not None:
        noisy_arr = ignore_epochs.as_array()
        for noisy_ind in range(noisy_arr.shape[0]):
            st = noisy_arr[noisy_ind, 0]
            en = noisy_arr[noisy_ind, 1]
            noisy_bool[np.where((time >= st) & (time <= en))[0]] = True

        # delta[noisy_timepoints] = np.nan
        # theta_ratio[noisy_timepoints] = np.nan
        # emg[noisy_timepoints] = np.nan

    # ----transform (if any) parameters such zscoring, minmax scaling etc. ----
    emg = np.log10(emg)
    thratio_sw_arr = np.vstack((theta_ratio, broadband_sw)).T

    # ---- Clustering---------

    emg_bool = np.zeros_like(emg).astype("bool")
    emg_bool[~noisy_bool] = gaussian_classify(emg[~noisy_bool]).astype("bool")

    nrem_rem_bool = np.zeros_like(emg).astype("bool")
    nrem_rem_bool[~noisy_bool & ~emg_bool] = gaussian_classify(
        thratio_sw_arr[~noisy_bool & ~emg_bool]
    ).astype("bool")

    # Using only theta ratio column to separate active and quiet awake
    aw_qw_bool = np.zeros_like(emg).astype("bool")
    aw_qw_label, aw_qw_fit_params = gaussian_classify(
        thratio_sw_arr[~noisy_bool & emg_bool][:, 0], ret_params=True
    )
    aw_qw_bool[~noisy_bool & emg_bool] = aw_qw_label.astype("bool")

    # --- states: Active wake (AW), Quiet Wake (QW), REM, NREM
    # initialize all states to Quiet wake
    states = np.array([""] * len(theta_ratio), dtype="U4")
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
        params = [broadband_sw, theta_ratio, emg]
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
        from bokeh.palettes import Plasma

        bplot.output_file(fp_bokeh_plot, title="Sleep scoring results")
        tools = "pan,box_zoom,reset"
        dimensions = dict(tools=tools, width=1000, height=200)
        line_kw = dict(line_width=2, color="black")
        p_sw = bplot.figure(**dimensions, y_axis_label="Arb. units")
        p_sw.line(time, broadband_sw, legend_label="broadband slow wave", **line_kw)

        p_theta = bplot.figure(x_range=p_sw.x_range, **dimensions)
        p_theta.line(time, theta_ratio, legend_label="theta_ratio", **line_kw)

        p_emg = bplot.figure(x_range=p_sw.x_range, **dimensions)
        p_emg.line(time, emg, legend_label="EMG", **line_kw)

        p_spect = bplot.figure(x_range=p_sw.x_range, **dimensions)
        p_spect.image(
            image=[theta_chan_sg.traces],
            x=0,
            y=0,
            dw=time[-1],
            dh=100,
            palette=Plasma[10],
            level="image",
        )

        p_rem_nrem = bplot.figure(
            title="NREM vs REM (low EMG)",
            width=400,
            height=400,
            x_axis_label="Theta ratio",
            y_axis_label="Broadband slow wave",
        )
        p_rem_nrem.circle(
            theta_ratio[~noisy_bool & ~emg_bool & nrem_rem_bool],
            broadband_sw[~noisy_bool & ~emg_bool & nrem_rem_bool],
            color="tomato",
            alpha=0.5,
            size=2,
            legend_label="REM",
        )
        p_rem_nrem.circle(
            theta_ratio[~noisy_bool & ~emg_bool & ~nrem_rem_bool],
            broadband_sw[~noisy_bool & ~emg_bool & ~nrem_rem_bool],
            color="blue",
            alpha=0.5,
            size=2,
            legend_label="NREM",
        )

        p_aw_qw = bplot.figure(
            title="Active vs Quiet wake (high EMG)",
            width=400,
            height=400,
            x_axis_label="Theta ratio",
            y_axis_label="Density",
        )
        thratio_aw_qw = theta_ratio[~noisy_bool & emg_bool]
        bins_aw_qw = np.linspace(thratio_aw_qw.min(), thratio_aw_qw.max(), 200)
        hist_aw_qw = np.histogram(thratio_aw_qw, bins_aw_qw, density=True)[0]
        means = aw_qw_fit_params["means"]
        covs = aw_qw_fit_params["covariances"]
        weights = aw_qw_fit_params["weights"]
        qw_fit = (
            stats.norm.pdf(
                bins_aw_qw[:-1], float(means[0][0]), np.sqrt(float(covs[0][0][0]))
            )
            * weights[0]
        )
        aw_fit = (
            stats.norm.pdf(
                bins_aw_qw[:-1], float(means[1][0]), np.sqrt(float(covs[1][0][0]))
            )
            * weights[1]
        )

        p_aw_qw.vbar(
            bins_aw_qw[:-1],
            bottom=0,
            width=np.diff(bins_aw_qw)[0] / 2,
            top=hist_aw_qw,
            color="#e5d02e",
            legend_label="Overall",
        )
        p_aw_qw.line(
            bins_aw_qw[:-1], qw_fit, color="gray", legend_label="Quiet", line_width=2
        )
        p_aw_qw.line(
            bins_aw_qw[:-1], aw_fit, color="black", legend_label="Active", line_width=2
        )
        # p_aw_qw.circle(
        #     theta_ratio[~noisy_bool & emg_bool & aw_qw_bool],
        #     broadband_sw[~noisy_bool & emg_bool & aw_qw_bool],
        #     color="black",
        #     alpha=0.5,
        #     size=2,
        #     legend_label="Active Wake",
        # )
        # p_aw_qw.circle(
        #     theta_ratio[~noisy_bool & emg_bool & ~aw_qw_bool],
        #     broadband_sw[~noisy_bool & emg_bool & ~aw_qw_bool],
        #     color="blue",
        #     alpha=0.5,
        #     size=2,
        #     legend_label="NREM",
        # )

        p_states = bplot.figure(x_range=p_sw.x_range, **dimensions)
        p_states.yaxis.visible = False
        y = 0
        colors = ["blue", "tomato", "gray", "black"]
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

        bplot.save(column(p_states, p_sw, p_theta, p_emg, row(p_rem_nrem, p_aw_qw)))

    return epochs
