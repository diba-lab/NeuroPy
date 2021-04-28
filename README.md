# NeuroPy
Modules for electrophysiology analysis using python.

### Overview
These modules are primarily developed for hippocampal recordings.

### Minimum requirements
* python 3.9
* Numpy 1.20.2
* Scipy 1.6.2


### Steps to follow before you start using modules:

   * Make sure your data folder has `.xml` and `.eeg` files.
   * Open the `.eeg` file in neuroscope and categorize bad recording channels as `skipped` and non-lfp channels as `discard` in neuroscope
   * Add `ModulesPath` to your pythonpath

### Quick example

```python
"""
Raster plot with corresponding raw LFP, ripple band and example ripple events
"""
figure = Fig()
fig, gs = figure.draw(grid=(6, 1), size=(8, 5))
for sess in sessions:
    period = [12672, 12679] # seconds
    ripples = sess.ripple.events
    ripples = ripples[(ripples.start > period[0]) & (ripples.start < period[1])]
    lfpmaze = sess.recinfo.geteeg(chans=sess.theta.bestchan, timeRange=period)
    ripple_lfp = sess.recinfo.geteeg(chans=sess.ripple.bestchans[4], timeRange=period)
    ripple_lfp = signal_process.filter_sig.ripple(ripple_lfp)
    lfp_t = np.linspace(period[0], period[1], len(lfpmaze)) - period[0]

    # ----- lfp plot --------------
    ax = plt.subplot(gs[0])
    ax.plot(lfp_t, lfpmaze, "k")
    ax.plot(
        ripples.peaktime - period[0], 2600 * np.ones(len(ripples)), "*", color="#f4835d"
    )
    ax.plot([0, 0], [0, 2500], "k", lw=3)  # lfp scale bar 2.5 mV
    ax.axis("off")
    ax.set_title("Raw LFP", loc="left")
    ax.annotate("Ripple events", xy=(0, 0.9), xycoords="axes fraction", color="#f4835d")

    # ------ ripple band plot -----------
    axripple = plt.subplot(gs[1], sharex=ax)
    axripple.plot(lfp_t, stats.zscore(ripple_lfp), "gray", lw=0.8)
    axripple.set_ylim([-12, 12])
    axripple.axis("off")
    axripple.set_title("Ripple band (150-250 Hz)", loc="left")

    # ----- raster plot -----------
    axraster = plt.subplot(gs[2:], sharex=ax)
    sess.spikes.plot_raster(
        # spikes=sess.spikes.pyr,
        period=period,
        ax=axraster,
        tstart=period[0],
        # color="hot_r",
        # sort_by_frate=True,
    )


```

![Example Image](images/raster.png)

### Citing this package
If you use NeuroPy in your research, please consider citing it.

```
@misc{neuropy2021,
    author       = {Bapun Giri, Nat Kinsky},
    title        = {{NeuroPy: Electrophysiology analysis using Python}},
    year         = 2020--2021,
    version      = {0.0.1},
    url          = {https://github.com/diba-lab/NeuroPy}
    }
```
