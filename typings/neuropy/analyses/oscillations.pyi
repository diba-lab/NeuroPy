"""
This type stub file was generated by pyright.
"""

from ..core import Epoch, ProbeGroup, Signal

"""
This type stub file was generated by pyright.
"""
def detect_hpc_slow_wave_epochs(signal: Signal, probegroup: ProbeGroup, freq_band=..., thresh=..., mindur=..., maxdur=..., mergedist=..., ignore_epochs: Epoch = ...):
    """Caculate delta events

    chan --> filter delta --> identify peaks and troughs within sws epochs only --> identifies a slow wave as trough to peak --> thresholds for 100ms minimum duration

    Parameters
    ----------
    chan : int
        channel to be used for detection
    freq_band : tuple, optional
        frequency band in Hz, by default (0.5, 4)
    """
    ...

def detect_ripple_epochs(signal: Signal, probegroup: ProbeGroup = ..., freq_band=..., thresh=..., mindur=..., maxdur=..., mergedist=..., ignore_epochs: Epoch = ...):
    ...

def detect_theta_epochs():
    ...

def detect_spindle_epochs(signal: Signal, probegroup: ProbeGroup, freq_band=..., thresh=..., mindur=..., maxdur=..., mergedist=..., ignore_epochs: Epoch = ..., method=...):
    ...

def detect_gamma_epochs():
    ...

class Gamma:
    """Events and analysis related to gamma oscillations"""
    def __init__(self, basepath) -> None:
        ...
    
    def get_peak_intervals(self, lfp, band=..., lowthresh=..., highthresh=..., minDistance=..., minDuration=...):
        """Returns strong theta lfp. If it has multiple channels, then strong theta periods are calculated from that channel which has highest area under the curve in the theta frequency band. Parameters are applied on z-scored lfp.

        Parameters
        ----------
        lfp : array like, channels x time
            from which strong periods are concatenated and returned
        lowthresh : float, optional
            threshold above which it is considered strong, by default 0 which is mean of the selected channel
        highthresh : float, optional
            [description], by default 0.5
        minDistance : int, optional
            minimum gap between periods before they are merged, by default 300 samples
        minDuration : int, optional
            [description], by default 1250, which means theta period should atleast last for 1 second

        Returns
        -------
        2D array
            start and end frames where events exceeded the set thresholds
        """
        ...
    
    def csd(self, period, refchan, chans, band=..., window=...):
        """Calculating current source density using laplacian method

        Parameters
        ----------
        period : array
            period over which theta cycles are averaged
        refchan : int or array
            channel whose theta peak will be considered. If array then median of lfp across all channels will be chosen for peak detection
        chans : array
            channels for lfp data
        window : int, optional
            time window around theta peak in number of samples, by default 1250

        Returns:
        ----------
        csd : dataclass,
            a dataclass return from signal_process module
        """
        ...
    

