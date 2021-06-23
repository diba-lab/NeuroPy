import numpy as np
from pathlib import Path


class PhyIO:
    def __init__(self, dirname: Path) -> None:
        pass

    def _parse_files(self):
        """Gets spike times from Phy (https://github.com/cortex-lab/phy) compatible files.
        If shanks are in separate folder, then folder should have subfolders with names Shank1, Shank2, Shank3 and so on.

        Parameters
        ----------
        folder : str
            folder where Phy files are present
        fileformat : str, optional
            [description], by default "diff_folder"
        """
        spktimes = None
        spkinfo = None

        clufolder = Path(folder)
        if fileformat == "diff_folder":
            nShanks = self._obj.nShanks
            sRate = self._obj.sampfreq
            spkall, info, shankID, template_waveforms = [], [], [], []
            for shank in range(1, nShanks + 1):
                shank_folder = clufolder / f"Shank{shank}"
                print(shank_folder)
                if shank_folder.is_dir():
                    spktime = np.load(shank_folder / "spike_times.npy")
                    cluID = np.load(shank_folder / "spike_clusters.npy")
                    spk_templates_id = np.load(shank_folder / "spike_templates.npy")
                    spk_templates = np.load(shank_folder / "templates.npy")
                    cluinfo = pd.read_csv(
                        shank_folder / "cluster_info.tsv", delimiter="\t"
                    )
                    goodCellsID = cluinfo.id[cluinfo["q"] < 10].tolist()
                    info.append(cluinfo.loc[cluinfo["q"] < 10])
                    shankID.extend(shank * np.ones(len(goodCellsID)))

                    for i in range(len(goodCellsID)):
                        clu_spike_location = np.where(cluID == goodCellsID[i])[0]
                        spkframes = spktime[clu_spike_location]
                        cell_template_id, counts = np.unique(
                            spk_templates_id[clu_spike_location], return_counts=True
                        )
                        spkall.append(spkframes / sRate)
                        template_waveforms.append(
                            spk_templates[cell_template_id[np.argmax(counts)]]
                            .squeeze()
                            .T
                        )

            spkinfo = pd.concat(info, ignore_index=True)
            spkinfo["shank"] = shankID
            spktimes = spkall

        if fileformat == "same_folder":
            nShanks = self._obj.nShanks
            sRate = self._obj.sampfreq
            changroup = self._obj.channelgroups

            spktime = np.load(clufolder / "spike_times.npy")
            cluID = np.load(clufolder / "spike_clusters.npy")
            spk_templates_id = np.load(clufolder / "spike_templates.npy")
            spk_templates = np.load(clufolder / "templates.npy")
            cluinfo = pd.read_csv(clufolder / "cluster_info.tsv", delimiter="\t")
            if "q" in cluinfo.keys():
                goodCellsID = cluinfo.id[cluinfo["q"] < 10].tolist()
                info = cluinfo.loc[cluinfo["q"] < 10]
            else:
                print(
                    'No labels "q" found in phy data - using good for now, be sure to label with ":l q #"'
                )
                goodCellsID = cluinfo.id[(cluinfo["group"] == "good")].tolist()
                info = cluinfo.loc[(cluinfo["group"] == "good")]

            peakchan = info["ch"]
            shankID = [
                sh + 1
                for chan in peakchan
                for sh, grp in enumerate(changroup)
                if chan in grp
            ]

            spkall, template_waveforms = [], []
            for i in range(len(goodCellsID)):
                clu_spike_location = np.where(cluID == goodCellsID[i])[0]
                spkframes = spktime[clu_spike_location]
                cell_template_id, counts = np.unique(
                    spk_templates_id[clu_spike_location], return_counts=True
                )
                spkall.append(spkframes / sRate)
                template_waveforms.append(
                    spk_templates[cell_template_id[np.argmax(counts)]].squeeze().T
                )

            info["shank"] = shankID
            spkinfo = info
            spktimes = spkall
            # self.shankID = np.asarray(shankID)

        spkinfo = spkinfo.reset_index()
        if save_allspikes:
            spikes_ = {
                "times": spktimes,
                "info": spkinfo.reset_index(),
                "allspikes": spktime,
                "allcluIDs": cluID,
                "templates": template_waveforms,
            }
        else:
            spikes_ = {
                "times": spktimes,
                "info": spkinfo.reset_index(),
                "templates": template_waveforms,
            }

        labels = np.empty(len(spktimes), dtype="object")
        labels[np.where(spkinfo.q < 4)[0]] = "pyr"
        labels[np.where(spkinfo.q == 8)[0]] = "intneur"
        labels[np.where(spkinfo.q == 6)[0]] = "mua"

        self.spiketrains = spikes_["times"]
        self.labels = labels
        self.shankids = spkinfo["shank"].to_numpy()
        self.ids = np.arange(0, len(self.spiketrains))
        self.waveforms = template_waveforms
        self.metadata = None

        self._check_neurons()
