from ..core import ProbeGroup
import numpy as np


def write_spyking_circus(prb: ProbeGroup, rmv_badchans=True, shanksCombine=False):
    """Creates .prb file for spyking circus in the basepath folder

    Parameters
    ----------
    rmv_badchans : bool
        if True then removes badchannels from the .prb file, by default True
    shanksCombine : bool, optional
        if True then all shanks are combined in same channel group, by default False
    """
    nShanks = prb.n_shanks
    nChans = prb.n_contacts 
    channelgroups = self._obj.channelgroups[:nShanks]
    circus_prb = (self._obj.files.filePrefix).with_suffix(".prb")
    coords = self.coords.set_index("chan")

    if rmv_badchans:
        channelgroups = self._obj.goodchangrp[:nShanks]

    with circus_prb.open("w") as f:
        f.write(f"total_nb_channels = {nChans}\n")
        f.write(f"radius = 120\n")
        f.write("channel_groups = {\n")

        if shanksCombine:

            chan_list = np.concatenate(channelgroups[:nShanks])
            f.write(f"1: {{\n")
            f.write(f"'channels' : {[int(_) for _ in chan_list]},\n")
            f.write("'graph' : [],\n")
            f.write("'geometry' : {\n")

            for i, shank in enumerate(channelgroups):
                if shank:
                    for chan in shank:
                        x, y = coords.loc[chan]
                        f.write(f"{chan}: [{x+i*300},{y+i*400}],\n")

                    f.write("\n")
            f.write("}\n")
            f.write("},\n")

            f.write("}\n")

        else:
            for i, shank in enumerate(channelgroups):
                if shank:
                    f.write(f"{i+1}: {{\n")
                    f.write(f"'channels' : {[int(_) for _ in shank]},\n")
                    f.write("'graph' : [],\n")
                    f.write("'geometry' : {\n")

                    for chan in shank:
                        x, y = coords.loc[chan]
                        f.write(f"{chan}: [{x+i*300},{y+i*400}],\n")

                    f.write("}\n")
                    f.write("},\n\n")

            f.write("}\n")

    print(".prb file created for Spyking Circus")
