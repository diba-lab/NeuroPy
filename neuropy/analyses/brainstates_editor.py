import ephyviewer
import pandas as pd
import numpy as np
from neuropy.utils import signal_process
from neuropy import core
from neuropy.utils.spectrogramviewer_custom import SpectrogramViewer


class StatesSource(ephyviewer.WritableEpochSource):
    """NOTE: there is a potential bug in the `ephyviewer` package which breaks even their own example code for
    implementing the TimeFreqViewer by improperly loading all default_params that are list types as empty strings.

    NRK bugfix: Add the NRK bugfix line to the ephyviewer.base.py file to reload default params properly.
    116    def make_param_controller(self):
    117         if self.with_user_dialog and self._ControllerClass:
    118             self.all_params.blockSignals(True)
    119             self.params_controller = self._ControllerClass(parent=self, viewer=self)
    120             self.make_params()  # NRK bugfix"""
    def __init__(
        self,
        epochs,
        possible_labels,
        filename,
        color_labels=None,
        channel_name="",
        restrict_to_possible_labels=False,
    ):

        self.filename = filename
        self.epochs: core.Epoch = epochs

        ephyviewer.WritableEpochSource.__init__(
            self,
            epoch=None,
            possible_labels=possible_labels,
            color_labels=color_labels,
            channel_name=channel_name,
            restrict_to_possible_labels=restrict_to_possible_labels,
        )

    def load(self):
        """
        Returns a dictionary containing the data for an epoch.
        Data is loaded from the CSV file if it exists; otherwise the superclass
        implementation in WritableEpochSource.load() is called to create an
        empty dictionary with the correct keys and types.
        The method returns a dictionary containing the loaded data in this form:
        { 'time': np.array, 'duration': np.array, 'label': np.array, 'name': string }
        """

        if self.epochs is not None:
            # if file already exists, load previous epoch
            data = self.epochs.to_dataframe()
            state_number_dict = {1: "nrem", 2: "rem", 3: "quiet", 4: "active"}
            data["name"] = data["label"].map(state_number_dict)

            epoch_labels = np.array([f" State{_}" for _ in data["label"]])
            epoch = {
                "time": self.epochs.starts,
                "duration": self.epochs.durations,
                "label": self.epochs.labels,
            }
        else:
            # if file does NOT already exist, use superclass method for creating
            # an empty dictionary
            epoch = super().load()

        return epoch

    def save(self):
        df = pd.DataFrame()
        df["start"] = np.round(self.ep_times, 6)  # round to nearest microsecond
        df["end"] = np.round(self.ep_times, 6) + np.round(
            self.ep_durations
        )  # round to nearest microsecond
        df["duration"] = np.round(self.ep_durations, 6)  # round to nearest microsecond
        state_number_dict = {"nrem": 1, "rem": 2, "quiet": 3, "active": 4, "nan": 5}
        df["name"] = self.ep_labels
        df["state"] = df["name"].map(state_number_dict)
        # df.sort_values(["time", "duration", "name"], inplace=True)
        df.sort_values(["start", "duration", "name"], inplace=True)
        # df.to_pickle(self.filename)
        df.to_csv(self.filename)


def editor(
    states: core.Epoch, sigs: core.Signal, emg: core.Signal = None, paradigm=None, spikes=None, filename=None
):
    states_source = StatesSource(
        states, ["nrem", "rem", "quiet", "active"], filename=filename
    )
    # you must first create a main Qt application (for event loop)
    # app = ephyviewer.mkQApp()

    sample_rate = sigs.sampling_rate
    sigs = sigs.traces.reshape(-1, 1)
    filtered_sig = signal_process.filter_sig.bandpass(
        sigs, lf=120, hf=250, ax=0, fs=1250
    )

    theta_filt_sig = signal_process.filter_sig.bandpass(
        sigs, lf=4, hf=12, ax=0, fs=1250
    )
    t_start = 0.0

    # --- Create the main window that can contain several viewers
    win = ephyviewer.MainViewer(debug=True, show_auto_scale=True)

    # ---- signal viewer ------
    view_traces = ephyviewer.TraceViewer.from_numpy(
        np.hstack((sigs, filtered_sig, theta_filt_sig)), sample_rate, t_start, "traces (raw, ripple, theta)"
    )
    view_traces.params["scale_mode"] = "by_channel"
    view_traces.auto_scale()
    win.add_view(view_traces)

    # ---- emg viewer ----
    if emg is not None:
        emg_sr = emg.sampling_rate
        emg_t_start = emg.t_start
        emg = emg.traces.reshape(-1, 1)
        view_emg = ephyviewer.TraceViewer.from_numpy(
            emg, emg_sr, emg_t_start, "emg"
        )
        view_emg.params["scale_mode"] = "real_scale"
        win.add_view(view_emg)

    # ----- brainstates viewer and encoder -----------
    source_sig = ephyviewer.InMemoryAnalogSignalSource(sigs, sample_rate, t_start)
    # create a viewer for the encoder itself
    view_states = ephyviewer.EpochEncoder(source=states_source, name="brainstates")
    view_states.params["background_color"] = "#ffffff"
    view_states.params["label_fill_color"] = "#ffffff"
    view_states.params["xsize"] = 3000
    view_states.params["vline_color"] = "#000000"
    view_states.by_label_params["label0", "color"] = "#536DFE"
    view_states.by_label_params["label1", "color"] = "#FF8A80"
    view_states.by_label_params["label2", "color"] = "#9E9E9E"
    view_states.by_label_params["label3", "color"] = "#424242"
    win.add_view(view_states)

    # ----- wavelet view --------
    view_wvlt = ephyviewer.TimeFreqViewer(source=source_sig, name="wvlt")
    view_wvlt.params["show_axis"] = True
    view_wvlt.params["colormap"] = "jet"
    view_wvlt.params["display_labels"] = False
    view_wvlt.params["timefreq", "deltafreq"] = 0.1
    view_wvlt.params["timefreq", "f_stop"] = 40
    view_wvlt.params["xsize"] = 60.0
    win.add_view(view_wvlt, split_with="brainstates", orientation="horizontal")

    # -------- fourier spectrogram -----------
    view_spect = SpectrogramViewer(
        source=source_sig, name="spectrogram", freq_lim=(0, 40)
    )
    view_spect.params["xsize"] = 500.0
    view_spect.params["colormap"] = "jet" #"Spectral_r"
    view_spect.params["display_labels"] = False
    view_spect.params["scalogram", "binsize"] = 2
    view_spect.params["scalogram", "overlapratio"] = 0.5
    view_spect.params["scalogram", "scale"] = "dB"
    view_spect.params["scalogram", "scaling"] = "spectrum"
    view_spect.params["vline_color"] = "#000000"
    win.add_view(view_spect, split_with="wvlt")

    # ----- spikes --------
    if spikes is not None:
        spk_id = np.arange(len(spikes))

        all_spikes = []
        for i, (t, id_) in enumerate(zip(spikes, spk_id)):
            all_spikes.append({"time": t, "name": f"Unit {i}"})

        spike_source = ephyviewer.InMemorySpikeSource(all_spikes=all_spikes)
        view_spect = ephyviewer.SpikeTrainViewer(source=spike_source)
        win.add_view(view_spect)

    if paradigm is not None:
        all_paradigms = []
        for l in paradigm.labels:
            d = paradigm[l]
            all_paradigms.append(
                dict(time=d.starts, duration=d.durations, label=d.labels, name=l)
            )

        paradigm_source = ephyviewer.InMemoryEpochSource(all_epochs=all_paradigms)
        view_paradigm = ephyviewer.EpochViewer(source=paradigm_source, name="paradigm")
        view_paradigm.params["xsize"] = 50000

        win.add_view(view_paradigm, split_with="brainstates")

    # show main window and run Qapp
    # win.show()
    # return win, app

    # app.exec_()
    return win

if __name__ == "__main__":
    print('test')
    pass