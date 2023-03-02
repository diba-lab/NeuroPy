import unittest
import numpy as np
# import the package
import sys, os
from pathlib import Path

# Add Neuropy to the path as needed
tests_folder = Path(os.path.dirname(__file__))

try:
    import neuropy
except ModuleNotFoundError as e:    
    root_project_folder = tests_folder.parent
    print('root_project_folder: {}'.format(root_project_folder))
    neuropy_folder = root_project_folder.joinpath('neuropy')
    print('neuropy_folder: {}'.format(neuropy_folder))
    sys.path.insert(0, str(root_project_folder))
finally:
    from neuropy.core import Position, Neurons
    from neuropy.analyses.placefields import PlacefieldComputationParameters



class TestPlacefieldsMethods(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
        
    # def test_pf(self):
        ## Unfinished.
        # t = np.linspace(0, 1, 240000)
        # y = np.sin(2 * np.pi * 12 * t) * 100

        # pos = Position(traces=y.reshape(1, -1), t_start=0, sampling_rate=120)

        # spktrns = []
        # for i in range(-100, 100, 30):
        #     indices = np.where((pos.x >= i) & (pos.x <= i + 20))[0]
        #     indices = np.random.choice(indices, 4000)
        #     spktrns.append(indices / 120)
        # spktrns = np.array(spktrns)

        # neurons = Neurons(spiketrains=spktrns, t_start=0, t_stop=2000)
        # pf1d = Pf1D(neurons=neurons, position=pos, speed_thresh=0.1, grid_bin=5)
        


    def test_computation_config_hashing(self):
        ## Hash testing:
        obj1 = PlacefieldComputationParameters(speed_thresh=15.0, grid_bin=None, smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5)
        obj2 = PlacefieldComputationParameters(speed_thresh=15.0, grid_bin=None, smooth=(1.0, 1.0), frate_thresh=0.2, time_bin_size=0.5)
        self.assertEqual(obj1, obj2, f'The hashes of two objects with the same values should be equal, but: hash(obj1): {hash(obj1)}, hash(obj2): {hash(obj2)}!')
    

if __name__ == '__main__':
    unittest.main()
    