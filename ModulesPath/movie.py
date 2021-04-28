import cv2
import numpy as np
import matplotlib.pyplot as plt


# import plotly.express as px


class tracking_movie:
    """Class to manipulate behavioral tracking movies."""

    def __init__(self, movie_path=None):
        assert movie_path is not None
        # if isinstance(basepath, Recinfo):
        #     self._obj = basepath
        # else:
        #     self._obj = Recinfo(basepath)

        self.initialize_movie(movie_path)
        self.get_nframes()
        self.get_frame_rate()

    def initialize_movie(self, movie_path):
        """Load in movies"""
        self.movie_path = movie_path
        if movie_path is not None:
            self.vidobj = cv2.VideoCapture(movie_path)
            self.get_nframes()
            self.get_frame_rate()

    def get_nframes(self):
        """Get # frames in a movie"""
        self.nframes = self.vidobj.get(cv2.CAP_PROP_FRAME_COUNT)

        return self.nframes

    def get_frame_rate(self):
        """Gets frame rate from movie"""
        self.fps = self.vidobj.get(cv2.CAP_PROP_FPS)

        return self.fps

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

    def plot_frame(
        self,
        frame_num=0,
        frame_time=None,
        ax=None,
        plotter="matplotlib",
        animal_location=None,
        window_size=140,
    ):
        """Plot a specific frame or time from a movie.  Use plotter='plotly' for
        Colab, otherwise 'matplotlib' is fine.  Use animal_location ([xloc, yloc])
        to zoom into area where the animal is automatically with specified
        window_size"""

        # First grab the frame
        frame_to_plot = self.grab_frame(frame_num, frame_time)

        # Set up string to label plot
        if frame_time is None:
            title_str = "Frame # " + str(frame_num)
        else:
            title_str = "Time = " + "{:.2f}".format(frame_time) + " seconds"

        # Set up area to zoom into if specified!
        if animal_location is not None:
            xlims = animal_location[0] + np.asarray([-window_size / 2, window_size / 2])
            ylims = animal_location[1] + np.asarray([-window_size / 2, window_size / 2])

            # Now plot and label!
        if plotter == "matplotlib":  # this doesn't work as well in Colab...
            if ax is None:  # Set up figure and axes
                fig = plt.figure(figsize=(25, 25))
                ax = plt.subplot()

            ax.imshow(frame_to_plot)  # plot
            ax.set_title(title_str)  # label plot

            if animal_location is not None:
                ax.set_xlim(xlims)
                ax.set_ylim(ylims[[1, 0]])

            plt.show()

        elif plotter == "plotly":  # Works better in Colab!
            fig = px.imshow(frame_to_plot)  # plot
            fig.update_layout(title=title_str)  # label
            fig.update_xaxes(showticklabels=False)  # Turn off axes to look nice
            fig.update_yaxes(showticklabels=False)

            if animal_location is not None:
                fig.update_xaxes(range=xlims)
                fig.update_yaxes(range=ylims[[1, 0]])

            fig.show()  # need this to actually show the figure

        return ax


def nearest(items, pivot):
    """Find nearest item in arrray items to pivot"""
    return min(items, key=lambda x: abs(x - pivot))
