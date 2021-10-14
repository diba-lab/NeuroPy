import numpy as np
from pathlib import Path
import pandas as pd


class PhyIO:
    def __init__(self, dirname: Path, included_groups=("mua", "good")) -> None:
        self.source_dir = dirname
        self.sampling_rate = None
        self.spiketrains = None
        self.waveforms = None
        self.peak_waveforms = None
        self.peak_channels = None
        self.included_groups = included_groups
        self._parse_folder()

    def _parse_folder(self):
        params = {}
        with (self.source_dir / "params.py").open("r") as f:
            for line in f:
                line_values = (
                    line.replace("\n", "")
                    .replace('r"', '"')
                    .replace('"', "")
                    .split("=")
                )
                params[line_values[0].strip()] = line_values[1].strip()

        self.sampling_rate = int(float(params["sample_rate"]))
        self.n_channels = int(params["n_channels_dat"])
        self.n_features_per_channel = int(params["n_features_per_channel"])

        spktime = np.load(self.source_dir / "spike_times.npy")
        clu_ids = np.load(self.source_dir / "spike_clusters.npy")
        spk_templates_id = np.load(self.source_dir / "spike_templates.npy")
        spk_templates = np.load(self.source_dir / "templates.npy")
        cluinfo = pd.read_csv(self.source_dir / "cluster_info.tsv", delimiter="\t")

        # if self.include_noise_clusters:
        #     cluinfo = cluinfo[
        #         cluinfo["group"].isin(["mua", "good", "noise"])
        #     ].reset_index(drop=True)
        # else:
        #     cluinfo = cluinfo[cluinfo["group"].isin(["mua", "good"])].reset_index(
        #         drop=True
        #     )

        cluinfo = cluinfo[cluinfo["group"].isin(self.included_groups)].reset_index(
            drop=True
        )
        if "id" not in cluinfo:
            print(
                "id column does not exist in cluster_info.tsv. Using cluster_id column."
            )
            cluinfo["id"] = cluinfo["cluster_id"]

        self.cluster_info = cluinfo.copy()
        self.amplitudes = cluinfo["amp"].values
        self.peak_channels = cluinfo["ch"].values
        self.shank_ids = cluinfo["sh"].values

        if not self.cluster_info.empty:
            spiketrains, template_waveforms = [], []
            for clu in cluinfo.itertuples():
                clu_spike_location = np.where(clu_ids == clu.id)[0]
                spkframes = spktime[clu_spike_location]
                cell_template_id, counts = np.unique(
                    spk_templates_id[clu_spike_location], return_counts=True
                )
                spiketrains.append(spkframes / self.sampling_rate)
                template_waveforms.append(
                    spk_templates[cell_template_id[np.argmax(counts)]].squeeze().T
                )

            self.spiketrains = np.array(spiketrains, dtype="object")
            self.waveforms = np.array(template_waveforms)
            self.peak_waveforms = [
                wav[np.argmax(np.max(wav, axis=1))] for wav in template_waveforms
            ]
