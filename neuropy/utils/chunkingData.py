import numpy as np
import matplotlib.pyplot as plt


def chunk(filename, Destfile, nChans, SampFreq, start_time, end_time):
    # Name of source file
    # filename = "source_file.dat"

    # Name of destination file
    # Destfile = "destination_file.dat"

    # nChans = 134  # number of channels in your dat file
    # SampFreq = 30000

    # # start and end time which you want to extract
    # start_time = 20  # from this time in seconds
    # end_time = 60 * 60  # duration of chunk

    # read required chunk from the source file
    b1 = np.memmap(
        filename,
        dtype="int16",
        offset=2 * nChans * int(SampFreq * start_time),
        mode="r",
        shape=nChans * int(SampFreq * end_time),
    )

    # allocates space for the file
    c = np.memmap(Destfile, dtype="int16", mode="w+", shape=(len(b1)))
    c[: len(b1)] = b1
    # del c

    # writes the data to that space
    d = np.memmap(Destfile, dtype="int16", mode="r+", shape=(len(b1)))


if __name__ == '__main__':
    filename = '/data2/Other Peoples Data/Bapun/RatS-Day2NSD-2020-11-27_10-22-29.eeg'
    destfile = '/data2/Other Peoples Data/Bapun/RatS-Day2NSD-2020-11-27_10-22-29_ontrack3.eeg'
    chunk(filename, destfile, 195, 1250, 2721.768, 3302.03)