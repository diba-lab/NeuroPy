import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pathlib import Path
import numpy as np


def get_good_frames(
    vid_mean: np.ndarray,
    thresh: float or int or list or np.ndarray = 5,
    method: str in ["abs", "diff", "both"] = "diff",
):
    """Identify frames with zero illumination (usually at start of triggered recordings
    :param: thresh - threshold below (for absolute method) or above which (for diff method) you reject frames
            must be a length = 2 array-like variable if using "both" method, 1st = "abs" thresh, 2nd = "diff" thresh
    :param: method: "abs" - remove any frames with mean value below the threshold
                    "diff" - remove any frames where the absolute change in intensity exceeds the threshold
                    "both" - combine both with logical AND

    :return: good_bool: boolean the length of vid_mean with good frames = True"""

    assert method in ["abs", "diff", "both"]

    if method == "both":
        assert (
            len(thresh) == 2
        ), 'Need two thresholds for "both" method: [abs_thresh, diff_thresh]'
        abs_thresh, diff_thresh = thresh
    else:
        assert (
            isinstance(thresh, int)
            or isinstance(thresh, float)
            or (isinstance(thresh, list) and len(thresh) == 1)
        )
        if method == "abs":
            abs_thresh, diff_thresh = thresh, np.nan
        elif method == "diff":
            abs_thresh, diff_thresh = np.nan, thresh

    good_bool_abs = vid_mean > abs_thresh
    good_bool1 = np.abs(np.diff(vid_mean)) < diff_thresh
    good_bool_diff = np.bitwise_and(
        np.append(good_bool1[0], good_bool1), np.append(good_bool1, good_bool1[-1])
    )

    if method == "abs":
        good_bool = good_bool_abs
    elif method == "diff":
        good_bool = good_bool_diff
    elif method == "both":
        good_bool = np.bitwise_and(good_bool_abs, good_bool_diff)

    return good_bool


class Mask:
    """Class to draw a mask over an imaging FOV for excluding things outside of GRIN lens, schmutz on lens, etc."""

    def __init__(self, fov: plt.axes or np.ndarray):

        assert isinstance(fov, Axes) or isinstance(fov, np.ndarray)
        if isinstance(fov, np.ndarray):
            _, self.ax = plt.axes()
        else:
            self.ax = fov

        self.points = []
        self.binding_id = plt.connect("motion_notify_event", self.on_move)
        plt.connect("button_press_event", self.on_click)

        plt.show()

    def on_move(self, event):
        # get the x and y pixel coords
        x, y = event.x, event.y
        if event.inaxes:
            ax = event.inaxes  # the axes instance
            print("data coords %f %f" % (event.xdata, event.ydata))

    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            if event.inaxes:
                #             points.append([event.x, event.y])
                self.points.append(
                    self.ax.transData.inverted().transform([event.x, event.y])
                )
                self.points_arr = np.array(self.points).T
                self.ax.plot(self.points_arr[0], self.points_arr[1], "r")
        elif event.button is MouseButton.RIGHT:
            print("disconnecting callback")
            plt.disconnect(self.binding_id)
            self.points_arr = np.array(self.points).T
            self.points_arr = np.append(
                self.points_arr, self.points_arr[:, 0, None], axis=1
            )
            self.ax.plot(self.points_arr[0], self.points_arr[1], "r")
