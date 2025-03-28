import numpy as np
import pandas as pd


def get_performance_2ab(
    df, min_trials_per_sess=None, roll_window=80, delta_prob=None, smooth=2
):
    """Get performance on two armed bandit task

    Parameters
    ----------
    df : csv file containing all data
        _description_
    min_trials_per_sess : _type_, optional
        sessions with more than this number of trials will excluded.
    roll_window : int, optional
        no.of sessions over which performance is calculated, by default 80
    roll_step : int, optional
        _description_, by default 40
    delta_prob : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """

    if delta_prob is not None:
        prob_diff = np.abs(df["rewprobfull1"] - df["rewprobfull2"])
        df = df[prob_diff >= delta_prob]

    session_id = df["session#"].to_numpy()
    unq_session_id, n_trials = np.unique(session_id, return_counts=True)
    is_choice_high = df["is_choice_high"].to_numpy()

    if min_trials_per_sess is not None:
        bad_sessions = unq_session_id[n_trials < min_trials_per_sess]
        n_trials = n_trials[n_trials >= min_trials_per_sess]
        bad_trials = np.isin(session_id, bad_sessions)

        is_choice_high = is_choice_high[~bad_trials]

    # converting into n_sessions x n_trials dataframe
    is_choice_high_per_session = pd.DataFrame(
        np.split(is_choice_high.astype(int), np.cumsum(n_trials)[:-1])
    )
    prob_correct_per_trial = is_choice_high_per_session.mean(axis=0)
    trial_x = np.arange(len(prob_correct_per_trial)) + 1

    perf = np.array([np.mean(arr) for arr in is_choice_high_per_session])

    sess_div_perf = is_choice_high_per_session.rolling(
        window=roll_window, closed="right", min_periods=2
    ).mean()[roll_window - 1 :: roll_window]

    sess_div_perf_arr = sess_div_perf.to_numpy()

    sess_div_perf_arr = gaussian_filter1d(sess_div_perf_arr, axis=1, sigma=smooth)

    return sess_div_perf_arr


def get_port_bias_2ab(df, min_trials=250):

    ntrials_by_session = get_trial_metrics(df)[0]

    delta_prob = df["rewprobfull1"] - df["rewprobfull2"]
    port_choice = df["port"].to_numpy()
    port_choice[port_choice == 2] = -1
    is_choice_high = df["is_choice_high"]
    max_diff = 80
    nbins = int(2 * max_diff / 10) + 1
    bins = np.linspace(-max_diff, max_diff, nbins)

    mean_choice = stats.binned_statistic(
        delta_prob, port_choice, bins=bins, statistic="mean"
    )[0]
    bin_centers = bins[:-1] + 5
    # mean_choice[bin_centers < 0] = -1 * mean_choice[bin_centers < 0]

    return mean_choice, bin_centers
