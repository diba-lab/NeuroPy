import cv2
import numpy as np
import matplotlib.pyplot as plt

from parsePath import Recinfo


class tracking_movie:
    """Class to manipulate behavioral tracking movies."""

    def __init__(self, movie_path=None):

        # if isinstance(basepath, Recinfo):
        #     self._obj = basepath
        # else:
        #     self._obj = Recinfo(basepath)

        self.initialize_movie(movie_path)

    def initialize_movie(self, movie_path):
        self.movie_path = movie_path
        if movie_path is not None:
            self.vidobj = cv2.VideoCapture(movie_path)
            self.get_nframes()
            self.get_frame_rate()

    def get_nframes(self):
        """Get # frames in a movie"""
        self.nframes = self.vidobj.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_frame_rate(self):
        """Gets frame rate from movie"""
        self.fps = self.vidobj.get(cv2.CAP_PROP_FPS)

    def grab_frame(self, frame_num=0, frame_time=None):
        """Get a specific frame or time from a movie"""
        if frame_time is not None:
            frame_num = np.round(self.fps * frame_time).astype("int")

        # update current frame #
        self.vidobj.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = self.vidobj.read()  # grab the frame

        return frame

    def get_current_frame(self):
        return self.vidobj.get(cv2.CAP_PROP_POS_FRAMES)

    def plot_frame(self, frame_num=0, frame_time=None, ax=None):
        """Plot a specific frame from a movie"""
        if ax is None:
            ax = plt.subplot()

        ax.imshow(self.grab_frame(frame_num, frame_time))
        if frame_time is None:
            ax.set_title("Frame # " + str(frame_num))
        else:
            ax.set_title("Time = " + "{:.2f}".format(frame_time) + " seconds")
