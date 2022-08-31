import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def get_good_frames(
    vid_mean: np.ndarray,
    thresh: float or int = 5,
    method: str in ["abs", "diff"] = "diff",
):
    """Identify frames with zero illumination (usually at start of triggered recordings
    :param: thresh - threshold below (for absolute method) or above which (for diff method) you reject frames
    :param: method: "abs" - remove any frames with mean value below the threshold
                    "diff" - remove any frames where the absolute change in intensity exceeds the threshold

    :return: good_bool: boolean the length of vid_mean with good frames = True"""

    assert method in ["abs", "diff"]

    if method == "abs":
        good_bool = vid_mean > thresh
    elif method == "diff":
        good_bool1 = np.abs(np.diff(vid_mean)) < thresh
        good_bool = np.bitwise_and(
            np.append(good_bool1[0], good_bool1), np.append(good_bool1, good_bool1[-1])
        )

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
