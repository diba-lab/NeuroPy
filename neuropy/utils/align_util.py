import numpy as np
import pandas as pd
import datetime


def align_data(time_ref, time_data, data, t_align_ref=None, t_align_data=None):
    """
    Aligns data in 'data_align' to time points in 'time_ref'. Default uses absolute datetime timestamps.
    :param time_ref: pandas series of datetime timestamps to align data to
    :param time: pandas series of datetime timestamps
    :param data: ndarray or timeseries of data the same shape as time to interpolate to time_ref timestamps
    :param t_align_ref: timestamp in reference data where you have a reference event, e.g. a TTL from a different system.
    Use 'start' if the TTL triggered recording start. None (default) = use timestamps to align directly.
    :param t_align_data: opposite of `t_align_ref` - time in time_data where your alignment event occurs. None = default.

    IMPORTANT NOTE: In the event you sent a TTL from some other system to the reference system and data system,
    use both.
    default=None

    :return: data_aligned
    """
    warn_limit = 1

    if t_align_ref is not None and t_align_data is not None:
        # Figure out common start time for t_align_ref
        if (
            t_align_ref == "start"
        ):  # set to start if event coincides with reference system start
            t_align_ref = time_ref.iloc[0]
        else:  # make sure input is valid if not None or 'start
            assert isinstance(
                t_align_ref, (pd.Timestamp, datetime.datetime)
            ), "t_align_ref must be pandas.Timestamp or datetime.datetime object"

        # Figure out common start time for t_align_data
        if t_align_data == "start":
            t_align_data = time_data.iloc[0]
        else:
            assert isinstance(
                t_align_data, (pd.Timestamp, datetime.datetime)
            ), "t_align_data must be pandas.Timestamp or datetime.datetime object"

        # Now adjust t_align so that both match up!
        align_delta = t_align_ref - t_align_data
        if np.abs(align_delta.total_seconds()) > warn_limit:
            print(
                "time_align shifted by more than "
                + str(warn_limit)
                + " seconds. Make sure alignment event times correct!"
            )
        time_data = time_data + align_delta
    else:
        print("No alignment events input, aligning by timestamps directly")

    # Now align data and timestamps
    data_aligned = np.interp(time_ref, time_data, data)
    time_aligned_sec = np.interp(
        time_ref, time_data, (time_data - time_data.iloc[0]).dt.total_seconds()
    )

    time_aligned = time_data.iloc[0] + pd.to_timedelta(time_aligned_sec, unit="sec")

    return time_aligned, data_aligned


def align_data_TTL(time_ref, time_align, data_align):
    pass
