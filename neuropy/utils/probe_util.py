from ..core import ProbeGroup
import numpy as np
from pathlib import Path


def write_spyking_circus(
    file: Path, prb: ProbeGroup, rmv_badchans=True, shanksCombine=False
):
    """Creates .prb file for spyking circus in the basepath folder

    Parameters
    ----------
    rmv_badchans : bool
        if True then removes badchannels from the .prb file, by default True
    shanksCombine : bool, optional
        if True then all shanks are combined in same channel group, by default False
    """

    nChans = prb.n_contacts
    prb_data = prb.to_dataframe().set_index("channel_id")
    channelgroups = prb.get_channels(groupby="shank")
    if rmv_badchans:
        channelgroups = prb.get_connected_channels(groupby="shank")

    with file.open("w") as f:
        f.write(f"total_nb_channels = {nChans}\n")
        f.write(f"radius = 120\n")
        f.write("channel_groups = {\n")

        if shanksCombine:

            chan_list = np.concatenate(channelgroups)
            f.write(f"1: {{\n")
            f.write(f"'channels' : {[int(_) for _ in chan_list]},\n")
            f.write("'graph' : [],\n")
            f.write("'geometry' : {\n")

            for i, shank in enumerate(channelgroups):
                if shank.any():
                    for chan in shank:
                        x, y = (
                            prb_data.loc[[chan]].x.values[0],
                            prb_data.loc[[chan]].y.values[0],
                        )
                        f.write(f"{chan}: [{x+i*300},{y+i*400}],\n")

                    f.write("\n")
            f.write("}\n")
            f.write("},\n")

            f.write("}\n")

        else:
            for i, shank in enumerate(channelgroups):
                if shank.any():
                    f.write(f"{i+1}: {{\n")
                    f.write(f"'channels' : {[int(_) for _ in shank]},\n")
                    f.write("'graph' : [],\n")
                    f.write("'geometry' : {\n")

                    for chan in shank:
                        x, y = (
                            prb_data.loc[[chan]].x.values[0],
                            prb_data.loc[[chan]].y.values[0],
                        )
                        f.write(f"{chan}: [{x+i*300},{y+i*400}],\n")

                    f.write("}\n")
                    f.write("},\n\n")

            f.write("}\n")

    print(f"{file.name} file created for Spyking Circus")
