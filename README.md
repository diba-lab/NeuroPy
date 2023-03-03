# NeuroPy [![DOI](https://zenodo.org/badge/426124562.svg)](https://zenodo.org/badge/latestdoi/426124562)
Package for analyzing ephys data using python.

### Overview
This package is primarily developed for hippocampal recordings, but can also be used for general ephys data.

### Minimum requirements
* python 3.9
* Numpy 1.20.2
* Scipy 1.6.2


### Steps to follow before you start using modules:

   * Make sure your data folder has `.xml` and `.eeg` files.
   * Open the `.eeg` file in neuroscope and categorize bad recording channels as `skipped` and non-lfp channels as `discard` in neuroscope


### Quick example

```python
"""
Raster plot with corresponding raw LFP, ripple band and example ripple events
"""
from neuropy.core import Neurons
from neuropy import plotting
spiketrains = np.array([np.sort(np.random.rand(_)) for _ in range(100,200)],dtype=object) 
neurons = Neurons(spiketrains,t_stop=1000)

plotting.plot_raster(neurons,color = 'jet')

```

![Example Image](images/raster.png)


### DVC (Data Version Control) Setup
DVC is used to access and version the files used in testing neuropy (excuting the unit tests in `/tests`).

##### Instalation:
It can be installed via poetry via: `poetry add dvc[gdrive] --dev`, but this adds a butch of uneeded libraries

Or installed locally on Windows via: `choco install dvc`

See also: https://dvc.org/doc/user-guide/how-to/setup-google-drive-remote#using-a-custom-google-cloud-project-recommended

```bash
dvc remote add --default diba_shared_gdrive gdrive://1DyUM5aOhE5kIKzc7OOkVyBltYjUbJ7tX/dvcstore
dvc pull # this will be fetch the files needed from Google Drive, first prompting you for your credentials
```

### Citing this package
If you use NeuroPy in your research, please consider citing as
```
@software{neuropy2022,
  author       = {Bapun Giri, Nat Kinsky, Pho Hale},
  title        = {NeuroPy: Electrophysiology analysis using Python},
  month        = jun,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {0.0.3},
  doi          = {10.5281/zenodo.6647463},
  url          = {https://doi.org/10.5281/zenodo.6647463}
}
```
