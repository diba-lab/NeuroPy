import numpy as np

class BinInfo(object):
    """docstring for BinInfo.
    Object to be returned by _bin_pos_nD
 
    Replaces the following implementation:
         bin_info_out_dict = {'mode':mode, 'xstep':xstep, 'xnum_bins':xnum_bins}
        if y is not None:
            # if at least 2D output, add the y-axis properties to the info dictionary
            bin_info_out_dict['ystep'], bin_info_out_dict['ynum_bins']  = ystep, ynum_bins
        else:
            ybin = None
            
            
    Properties:
        mode = 'num_bins' ## Binning with Fixed Number of Bins:
        mode = 'bin_size' ## Binning with Fixed Bin Sizes:
 
     """
    def __init__(self, mode, xstep, xnum_bins, ystep=None, ynum_bins=None):
        super(BinInfo, self).__init__()
        self.mode = mode
        self.xstep = xstep
        self.xnum_bins = xnum_bins
        self.ystep = ystep
        self.ynum_bins = ynum_bins
        
    def setup(self):
        pass
  

    

class BinnedPositionsMixin(object):
    """docstring for BinnedPositionsMixin."""

    @property
    def xbin_centers(self):
        """ the x-position of the centers of each xbin. Note that there is (n_xbins - 1) of these. """
        return self.xbin[:-1] + np.diff(self.xbin) / 2

    @property
    def ybin_centers(self):
        """ the y-position of the centers of each xbin. Note that there is (n_ybins - 1) of these. """
        if self.ybin is None:
            return None
        else:
            return self.ybin[:-1] + np.diff(self.ybin) / 2

    @property
    def xbin_labels(self):
        """ the labels of each xbin center. Starts at 1!"""
        return np.arange(start=1, stop=len(self.xbin)) # bin labels are 1-indexed, thus adding 1

    @property
    def ybin_labels(self):
        """ the labels of each ybin center. Starts at 1!"""
        if self.ybin is None:
            return None
        else:
            return np.arange(start=1, stop=len(self.ybin))


