"""
This type stub file was generated by pyright.
"""

from ..core import Epoch, Signal

def plot_epochs(ax, epochs: Epoch, ymin=..., ymax=..., color=..., style=...):
    """Plots epochs on a given axis, with different style of plotting

    Parameters
    ----------
    ax : axis
        [description]
    epochs : [type]
        [description]
    ymin : float, optional
        [description], by default 0.5
    ymax : float, optional
        [description], by default 0.55
    color : str, optional
        [description], by default "gray"

    Returns
    -------
    [type]
        [description]
    """
    ...

def plot_hypnogram(epochs: Epoch, ax=..., unit=..., collapsed=..., annotate=...): # -> Any:
    """Plot hypnogram

    Parameters
    ----------
    ax : [type], optional
        axis to plot onto, by default None
    tstart : float, optional
        start of hypnogram, by default 0.0, helps in positioning of hypnogram
    unit : str, optional
        unit of timepoints, 's'=seconds or 'h'=hour, by default "s"
    collapsed : bool, optional
        if true then all states have same vertical spans, by default False and has classic look

    Returns
    -------
    [type]
        [description]

    """
    ...

def plot_epochs_with_raster(self, ax=...): # -> None:
    ...

def plot_artifact_epochs(epochs: Epoch, signal: Signal, downsample_factor: int = ...): # -> Any:
    """Plots artifact epochs against a signal

    Parameters
    ----------
    epochs : Epoch
        [description]
    signal : Signal
        [description]
    downsample_factor : int, optional
        It is much faster to plot downsampled signal, by default 5

    Returns
    -------
    [type]
        [description]
    """
    ...
