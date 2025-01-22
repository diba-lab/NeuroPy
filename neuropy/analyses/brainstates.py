import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from pathlib import Path
import typing

from .. import core
from ..utils import mathutil, signal_process


def emg_from_LFP(
    signal: core.Signal,
    window,
    overlap=0.0,
    probe: core.ProbeGroup = None,
    min_dist=0,
    as_signal=False,
    n_jobs=1,
):
    """Estimates muscle activity using power in high frequency band (300,600 Hz).

    Method:
    LFP --> bandpass filtered (300-600 Hz) --> Pearson correlation across pairs of channels --> Mean of pearson correlations --> EMG activity

    Note: Prior to estimation, it is advised to visualize LFP power spectrum and confirm they don't have sharp peaks from artifacts in 300-600 Hz band. They can drastically affect the EMG estimation.

    Parameters
    ----------
    signal : core.Signal object
        signal containing LFP traces
    window : float
        window size in seconds in which correlations are computed
    overlap : float,
        overlap between adjacent window in seconds, by default 0
    probe : core.Probegroup
        channel mapping of the signal object
    min_dist:
        if probe is provided, use only channels that are separated by at least this much distance, in um
    as_signal: bool
        whether to return emg as a signal object
    n_jobs: int,
        number of cpu/processes to use

    Returns
    -------
    array or core.Signal
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

        # yf1 = signal_process.filter_sig.bandpass(lfp_req, lf=315, hf=485, fs=sRate)
        # yf2 = signal_process.filter_sig.bandpass(lfp_req, lf=520, hf=600, fs=sRate)
        # yf = yf1 + yf2

        # yf = signal_process.filter_sig.highpass(lfp_req, cutoff=300, fs=sRate)

        # ltriang = np.tril_indices(n_emg_chans, k=-1)
        ltriang = np.tril(pairs_bool, k=-1)
        return np.corrcoef(yf)[ltriang].mean()

    corr_per_window = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(corrchan)(start) for start in timepoints
    )
    emg_lfp = np.asarray(corr_per_window)

    print("emg calculation done")

    if as_signal:
        fs = 1 / (window - overlap)
        return core.Signal(
            traces=emg_lfp[np.newaxis, :], sampling_rate=fs, t_start=window / 2
        )
    else:
        return emg_lfp


def detect_brainstates_epochs(
    signal: core.Signal,
    theta_channel: int,
    delta_channel: int,
    probe: core.ProbeGroup,
    window=2,
    overlap=1,
    sigma=4,
    emg_signal=None,
    ignore_epochs: core.Epoch = None,
    threshold_type: typing.Literal["default", "schmitt"] = "default",
    # plot=False,
    fp_bokeh_plot: typing.Union[Path, str] = None,
):
    """detects sleep states using LFP.

    Parameters
    ----------
    signal : core.Signal object
        Signal object containing LFP traces across all good channels preferrable
    theta_channel: bool, optional
        specify channel_id that can be used for theta power calculation, by None
    delta_channel: bool, optional
        specify channel that can be used for slow wave calculation, by None
    probe: core.Probe object
        probemap containing channel coordinates and ids of good and bad channels
    window : float, optional
        window size for spectrogram calculation, by default 2 seconds
    overlap : float, optional
        by how much the adjacent windows overlap, by default 1 seconds
    sigma : float, optional
        smoothing window in seconds, by default 4 seconds
    emg_signal : bool, optional
        if emg has already been calculated pass that as a signal object, by default None, then emg is estimated using correlation EMG
    min_dur: bool, optional
        minimum duration of each state, by None
    ignore_epochs: bool, optional
        epochs which are ignored during scoring, could be noise epochs, by None
    fp_bokeh_plot: Path to file, optional
        if given then .html file is saved detailing some of scoring parameters and classification, by default None
    """

    # freqs = np.geomspace(1, 100, 100)
    spect_kw = dict(window=window, overlap=overlap, norm_sig=True)

    smooth_ = lambda arr: gaussian_filter1d(arr, sigma=sigma / (window - overlap))
    print(f"channel for sleep detection: {theta_channel,delta_channel}")

    # ---- theta-delta ratio calculation -----
    if theta_channel is not None:
        theta_signal = signal.time_slice(channel_id=[theta_channel])
        theta_chan_sg = signal_process.FourierSg(theta_signal, **spect_kw)

    if delta_channel is not None:
        if delta_channel == theta_channel:
            delta_chan_sg = theta_chan_sg
        else:
            delta_signal = signal.time_slice(channel_id=[delta_channel])
            delta_chan_sg = signal_process.FourierSg(delta_signal, **spect_kw)

    time = theta_chan_sg.time
    dt = window - overlap

    print(f"spectral properties calculated")

    # ---- emg processing ----
    if emg_signal is None:
        # emg_kw = dict(window=window, overlap=overlap)
        emg_kw = dict(window=1, overlap=0.0)
        emg = emg_from_LFP(signal=signal, probe=probe, **emg_kw)
        emg = gaussian_filter1d(emg, sigma=20)
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
        theta_chan_sg.get_noisy_spect_bool(thresh=5),
        delta_chan_sg.get_noisy_spect_bool(thresh=5),
    )
    noisy_bool = np.logical_or(noisy_bool, noisy_spect_bool)

    # ------ features to be used for scoring ---------
    theta = theta_chan_sg.get_band_power(5, 10)
    theta[~noisy_bool] = smooth_(theta[~noisy_bool])
    bp_2to16 = theta_chan_sg.get_band_power(2, 16)
    bp_2to16[~noisy_bool] = smooth_(bp_2to16[~noisy_bool])

    delta = delta_chan_sg.get_band_power(1, 4)
    delta[~noisy_bool] = smooth_(delta[~noisy_bool])

    theta_dominance = theta / bp_2to16  # buzsaki lab, better for high and low theta
    theta_delta_ratio = theta / delta  # used for REM vs NREM

    # Usually the first principal component has highiest weights in lower frequency band (1-32 Hz)
    # broadband_sw = np.zeros_like(theta)
    # broadband_sw[~noisy_bool] = smooth_(
    #     PCA(n_components=1)
    #     .fit_transform(delta_chan_zscored_spect[:, ~noisy_bool].T)
    #     .squeeze()
    # )

    # ----transform (if any) parameters such zscoring, minmax scaling etc. ----
    emg[np.isnan(emg)] = np.nanmax(emg)
    emg = np.log10(emg)
    # thdom_bsw_arr = np.vstack((theta_dominance, broadband_sw)).T

    # ---- Clustering---------
    classify = lambda x: mathutil.bimodal_classify(
        x, ret_params=True, threshold_type=threshold_type
    )
    emg_bool = np.zeros_like(emg).astype("bool")
    emg_labels, emg_fit_params = classify(emg[~noisy_bool])
    emg_bool[~noisy_bool] = emg_labels.astype("bool")

    nrem_rem_bool = np.zeros_like(emg).astype("bool")
    nrem_rem_labels, nrem_rem_fit_params = classify(
        theta_delta_ratio[~noisy_bool & ~emg_bool]
    )
    nrem_rem_bool[~noisy_bool & ~emg_bool] = nrem_rem_labels.astype("bool")

    # Using only theta dominance to separate active and quiet awake
    aw_qw_bool = np.zeros_like(emg).astype("bool")
    aw_qw_label, aw_qw_fit_params = classify(theta_dominance[~noisy_bool & emg_bool])
    aw_qw_bool[~noisy_bool & emg_bool] = aw_qw_label.astype("bool")

    # --- states: Active wake (AW), Quiet Wake (QW), REM, NREM
    # initialize empty label states
    states = np.array([""] * len(time), dtype="U5")
    states[noisy_bool] = "NOISE"
    states[~noisy_bool & emg_bool & aw_qw_bool] = "AW"
    states[~noisy_bool & emg_bool & ~aw_qw_bool] = "QW"
    states[~noisy_bool & ~emg_bool & nrem_rem_bool] = "REM"
    states[~noisy_bool & ~emg_bool & ~nrem_rem_bool] = "NREM"

    # ---- Refining states --------
    # removing REM which happens within long WAKE. If REM follows after 200s of WAKE, then change them to Quiet waking.
    for rem_indx in np.where(states == "REM")[0]:
        n_indx_before = 200 // dt  # 200 seconds window
        start_indx = np.max([0, rem_indx - n_indx_before])
        wk_bool = np.isin(states[start_indx:rem_indx], ["AW", "QW", "NOISE"])
        if wk_bool.sum() >= n_indx_before:
            states[rem_indx] = "QW"

    # --- TODO micro-arousals ---------
    # ma_bool = emg_bool & delta_bool
    # states[ma_indices] = "MA"

    # ---- make epochs -----
    epochs = core.Epoch.from_string_array(states, t=time)
    metadata = {
        "window": window,
        "overlap": overlap,
        "theta_channel": theta_channel,
        "delta_channel": delta_channel,
    }
    epochs.metadata = metadata

    # epochs = epochs.duration_slice(min_dur=min_dur)
    # epochs = epochs.fill_blank("from_right")  # this will also fill ignore_epochs

    # clearing out ignore_epochs
    # if ignore_epochs is not None:
    #     for e in ignore_epochs.as_array():
    #         epochs = epochs.delete_in_between(e[0], e[1])

    # if plot:
    #     _, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True)
    #     params = [broadband_sw, theta_delta_ratio, emg]
    #     params_names = ["Delta", "Theta ratio", "EMG"]

    #     for i, (param, name) in enumerate(zip(params, params_names)):
    #         axs[i].plot(time, param, "k.", markersize=1)
    #         # axs[0].set_ylim([-1, 4])
    #         axs[i].set_title(name, loc="left")

    #     states_colors = dict(NREM="#e920e2", REM="#f7abf4", QW="#fbc77e", AW="#e28708")
    #     plot_epochs(
    #         epochs=epochs,
    #         ax=axs[3],
    #         # labels_order=["NREM", "REM", "QW", "AW"],
    #         # colors=states_colors,
    #     )

    if fp_bokeh_plot is not None:
        try:
            import bokeh.plotting as bplot
            from bokeh.models import Range1d
            from bokeh.layouts import column, row
        except:
            raise ImportError("Bokeh is not installed")

        bplot.output_file(fp_bokeh_plot, title="Sleep scoring results")
        tools = "pan,box_zoom,reset"
        dimensions = dict(tools=tools, width=1000, height=180)

        def plot_feature(x, y, title=None):
            feature_kw = dict(color="black", line_width=2, alpha=0.7)
            p = bplot.figure(**dimensions, y_axis_label="Amplitude", title=title)
            p.line(x[~noisy_bool], y[~noisy_bool], **feature_kw)
            return p

        p_sw = plot_feature(time, delta, f"Delta activity (Channel:{delta_channel})")
        p_sw.y_range = Range1d(
            delta[~noisy_bool].min(), stats.scoreatpercentile(delta[~noisy_bool], 99.9)
        )
        p_theta = plot_feature(
            time, theta_dominance, f"Theta activity (Channel:{theta_channel})"
        )
        p_theta_delta_ratio = plot_feature(time, theta_delta_ratio, "Theta delta ratio")
        p_theta.x_range = p_sw.x_range
        p_theta_delta_ratio.x_range = p_sw.x_range
        p_emg = plot_feature(time, emg, "EMG activity")
        p_emg.x_range = p_sw.x_range

        def plot_thresh(x, fit_params, x_label, low_label, high_label, title):
            p = bplot.figure(
                title=title,
                x_axis_label=x_label,
                y_axis_label="Density",
                width=330,
                height=330,
            )
            hist, bins = np.histogram(x, 200, density=True)
            means = fit_params["means"]
            covs = fit_params["covariances"]
            weights = fit_params["weights"]
            get_fit = lambda x, mu, v, w: stats.norm.pdf(x, mu, np.sqrt(v)) * w
            lowfit = get_fit(bins[:-1], means[0], covs[0], weights[0])
            highfit = get_fit(bins[:-1], means[1], covs[1], weights[1])

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
            x_label="EMG activity",
            low_label="Sleep",
            high_label="Wake",
            title="Sleep vs Wake",
        )

        p_nrem_rem_dist = plot_thresh(
            theta_delta_ratio[~noisy_bool & ~emg_bool],
            fit_params=nrem_rem_fit_params,
            x_label="Theta delta ratio",
            low_label="NREM",
            high_label="REM",
            title="NREM vs REM (low EMG)",
        )

        p_aw_qw_dist = plot_thresh(
            theta_dominance[~noisy_bool & emg_bool],
            fit_params=aw_qw_fit_params,
            x_label="Theta activity",
            low_label="Quiet",
            high_label="Active",
            title="Active vs Quiet wake (high EMG)",
        )

        p_states = bplot.figure(x_range=p_sw.x_range, title="Brainstates", **dimensions)
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
                p_theta_delta_ratio,
                p_emg,
                row(p_emg_dist, p_nrem_rem_dist, p_aw_qw_dist),
            )
        )
        print(f"{fp_bokeh_plot} saved")

    return epochs