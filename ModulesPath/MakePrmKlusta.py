import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from parsePath import Recinfo

# folderPath = '../'


class makePrmPrb:
    def __init__(self, basepath):
        if isinstance(basepath, Recinfo):
            self._obj = basepath
        else:
            self._obj = Recinfo(basepath)
        # print(self._obj.makePrmPrb.prmTemplate)
        self.prmTemplate = (
            "/home/bapung/Documents/MATLAB/pythonprogs/RoutineAnalysis/template.prm"
        )
        self.prbTemplate = (
            "/home/bapung/Documents/MATLAB/pythonprogs/RoutineAnalysis/template.prb"
        )

        # self._myinfo = recinfo(basePath)

        # recInfo = np.load(self._files.basics, allow_pickle=True)
        # self.sampFreq = recInfo.item().get("sRate")
        # self.chan_session = recInfo.item().get("channels")
        # self.nChansDat = recInfo.item().get("nChans")
        # self.channelgroups = recInfo.item().get("channelgroups")
        # self.nShanks = recInfo.item().get("nShanks")

    def makePrm(self):
        for shank in range(1, self.nShanks + 1):
            with open(self.prmTemplate) as f:
                if not os.path.exists(self.basePath + "Shank" + str(shank)):
                    os.mkdir(self.basePath + "Shank" + str(shank))
                outfile_prm = Path(
                    self.basePath,
                    "Shank" + str(shank),
                    self.sessionName + "sh" + str(shank) + ".prm",
                )

                with outfile_prm.open("w") as f1:
                    for line in f:

                        if "experiment_name" in line:

                            f1.write(
                                "experiment_name = '"
                                + self.sessionName
                                + "sh"
                                + str(shank)
                                + "'\n"
                            )
                        elif "prb_file" in line:
                            f1.write(
                                "prb_file = '"
                                + str(outfile_prm.with_suffix(".prb"))
                                + "'\n"
                            )
                        elif "raw_data_files" in line:
                            f1.write(
                                "   raw_data_files = ['" + self.filePrefix + ".dat'],\n"
                            )
                        elif "sample_rate" in line:
                            f1.write("   sample_rate = " + str(self.sampFreq) + ",\n")
                        elif "n_channels" in line:
                            f1.write("  n_channels = " + str(self.nChansDat) + ",\n")

                        else:
                            f1.write(line)

    def makePrmServer(self):

        self.serverbasePath = Path("/nfs/turbo/umms-kdiba/Bapun/")
        self.serverPath = self.serverbasePath / self.name / self.day

        for shank in range(1, self.nShanks + 1):
            with open(self.prmTemplate) as f:
                if not os.path.exists(Path(self.serverPath, "Shank" + str(shank))):
                    os.mkdir(Path(self.serverPath, "Shank" + str(shank)))
                outfile_prm = Path(
                    self.serverPath,
                    "Shank" + str(shank),
                    self.sessionName + "sh" + str(shank) + ".prm",
                )
                # print(outfile_prm.suffix)

                with outfile_prm.open("w") as f1:
                    for line in f:

                        if "experiment_name" in line:

                            f1.write(
                                "experiment_name = '"
                                + self.sessionName
                                + "sh"
                                + str(shank)
                                + "'\n"
                            )
                        elif "prb_file" in line:
                            f1.write(
                                "prb_file = '"
                                + str(outfile_prm.with_suffix(".prb"))
                                + "'\n"
                            )
                        elif "raw_data_files" in line:
                            f1.write(
                                "   raw_data_files = ['"
                                + str(Path(self.serverPath, self.subname))
                                + ".dat'],\n"
                            )
                        elif "sample_rate" in line:
                            f1.write("   sample_rate = " + str(self.sampFreq) + ",\n")
                        elif "n_channels" in line:
                            f1.write("  n_channels = " + str(self.nChansDat) + ",\n")

                        else:
                            f1.write(line)

    def makePrb(self):
        for shank in range(1, self.nShanks + 1):

            chan_list = self.channelgroups[shank - 1]
            with open(self.prbTemplate) as f:
                if not os.path.exists(Path(self.basePath, "Shank" + str(shank))):
                    os.mkdir(self.basePath + "Shank" + str(shank))
                outfile_prefix = Path(
                    self.basePath,
                    "Shank" + str(shank),
                    self.sessionName + "sh" + str(shank) + ".prb",
                )

                with outfile_prefix.open("w") as f1:
                    for line in f:

                        if "Shank index" in line:
                            f1.write("# Shank index. \n")
                            f1.write(str(shank - 1) + ":\n")
                            next(f)

                        elif "channels" in line:
                            f1.write("'channels' : " + str(chan_list) + ",")

                        elif "graph" in line:
                            f1.write("'graph' : [\n")
                            for i, chan in enumerate(chan_list[:-2]):
                                f1.write(f"({chan},{chan_list[i + 1]}),\n")
                                f1.write(f"({chan},{chan_list[i + 2]}),\n")

                            f1.write(f"({chan_list[-2]},{chan_list[-1]}),\n")
                            for i in range(13):
                                next(f)

                        elif "geometry" in line:
                            f1.write("'geometry' : {\n")
                            # f1.write("(" +str(chan_list[0])',' +")")
                            chan_height = np.arange(320, 10, -20)
                            for i in range(16):
                                f1.write(
                                    str(chan_list[i])
                                    + ":"
                                    + str((0, chan_height[i]))
                                    + ",\n"
                                )

                            for i in range(8):
                                next(f)

                        else:
                            f1.write(line)

    def makePrbServer(self):

        self.serverbasePath = Path("/nfs/turbo/umms-kdiba/Bapun/")
        self.serverPath = self.serverbasePath / self.name / self.day

        for shank in range(1, self.nShanks + 1):

            chan_list = self.channelgroups[shank - 1]
            with open(self.prbTemplate) as f:
                if not os.path.exists(Path(self.serverPath, "Shank" + str(shank))):
                    os.mkdir(Path(self.serverPath, "Shank" + str(shank)))
                outfile_prefix = Path(
                    self.serverPath,
                    "Shank" + str(shank),
                    self.sessionName + "sh" + str(shank) + ".prb",
                )

                with outfile_prefix.open("w") as f1:
                    for line in f:

                        if "Shank index" in line:
                            f1.write("# Shank index. \n")
                            f1.write(str(shank - 1) + ":\n")
                            next(f)

                        elif "channels" in line:
                            f1.write("'channels' : " + str(chan_list) + ",")

                        elif "graph" in line:
                            f1.write("'graph' : [\n")
                            for i, chan in enumerate(chan_list[:-2]):
                                f1.write(f"({chan},{chan_list[i + 1]}),\n")
                                f1.write(f"({chan},{chan_list[i + 2]}),\n")

                            f1.write(f"({chan_list[-2]},{chan_list[-1]}),\n")

                            for i in range(13):
                                next(f)

                        elif "geometry" in line:
                            f1.write("'geometry' : {\n")
                            # f1.write("(" +str(chan_list[0])',' +")")
                            chan_height = np.arange(320, 10, -20)
                            for i in range(16):
                                f1.write(
                                    str(chan_list[i])
                                    + ":"
                                    + str((0, chan_height[i]))
                                    + ",\n"
                                )

                            for i in range(8):
                                next(f)

                        else:
                            f1.write(line)

    def makePrbCircus(self, probetype, shanksCombine=0):
        nShanks = self._obj.nShanks
        nChans = self._obj.nChans
        channelgroups = self._obj.channelgroups
        circus_prb = (self._obj.files.filePrefix).with_suffix(".prb")
        if probetype == "buzsaki":
            xpos = [0, 37, 4, 33, 8, 29, 12, 20]
            ypos = np.arange(160, 0, -20)
        elif probetype == "diagbio":
            xpos = [16 * (_ % 2) for _ in range(16)]
            ypos = [15 * 16 - _ * 15 for _ in range(16)]
        with circus_prb.open("w") as f:
            f.write(f"total_nb_channels = {nChans}\n")
            f.write(f"radius = 120\n")
            f.write("channel_groups = {\n")

            if shanksCombine:

                chan_list = np.concatenate(channelgroups[:nShanks])
                f.write(f"1: {{\n")
                f.write(f"'channels' : {list(chan_list)},\n")
                f.write("'graph' : [],\n")
                f.write("'geometry' : {\n")

                for shank in range(1, nShanks + 1):
                    for chan, x, y in zip(channelgroups[shank - 1], xpos, ypos):
                        f.write(f"{chan}: [{x+(shank-1)*300},{y+(shank-1)*400}],\n")

                    f.write("\n")
                f.write("}\n")
                f.write("},\n")

                f.write("}\n")

            else:
                for shank in range(1, nShanks + 1):
                    chan_list = channelgroups[shank - 1]

                    f.write(f"{shank}: {{\n")
                    f.write(f"'channels' : {chan_list},\n")
                    f.write("'graph' : [],\n")
                    f.write("'geometry' : {\n")

                    for chan, x, y in zip(chan_list, xpos, ypos):
                        f.write(f"{chan}: [{x+(shank-1)*300},{y+(shank-1)*400}],\n")

                    f.write("}\n")
                    f.write("},\n")

                f.write("}\n")

        print(".prb file created for Spyking Circus")
