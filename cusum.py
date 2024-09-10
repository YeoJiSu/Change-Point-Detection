# https://github.com/BMClab/BMC/blob/master/functions/detect_cusum.py
"""Cumulative sum algorithm (CUSUM) to detect abrupt changes in data.

Parameters
----------
x : 1D array_like
    data.
threshold : positive number, optional (default = 1)
    amplitude threshold for the change in the data.
drift : positive number, optional (default = 0)
    drift term that prevents any change in the absence of change.
ending : bool, optional (default = False)
    True (1) to estimate when the change ends; False (0) otherwise.
show : bool, optional (default = True)
    True (1) plots data in matplotlib figure, False (0) don't plot.
ax : a matplotlib.axes.Axes instance, optional (default = None).

Returns
-------
ta : 1D array_like [indi, indf], int
    alarm time (index of when the change was detected).
tai : 1D array_like, int
    index of when the change started.
taf : 1D array_like, int
    index of when the change ended (if `ending` is True).
amp : 1D array_like, float
    amplitude of changes (if `ending` is True).

Notes
-----
Tuning of the CUSUM algorithm according to Gustafsson (2000)[1]_:
Start with a very large `threshold`.
Choose `drift` to one half of the expected change, or adjust `drift` such
that `g` = 0 more than 50% of the time.
Then set the `threshold` so the required number of false alarms (this can
be done automatically) or delay for detection is obtained.
If faster detection is sought, try to decrease `drift`.
If fewer false alarms are wanted, try to increase `drift`.
If there is a subset of the change times that does not make sense,
try to increase `drift`.

Note that by default repeated sequential changes, i.e., changes that have
the same beginning (`tai`) are not deleted because the changes were
detected by the alarm (`ta`) at different instants. This is how the
classical CUSUM algorithm operates.

If you want to delete the repeated sequential changes and keep only the
beginning of the first sequential change, set the parameter `ending` to
True. In this case, the index of the ending of the change (`taf`) and the
amplitude of the change (or of the total amplitude for a repeated
sequential change) are calculated and only the first change of the repeated
sequential changes is kept. In this case, it is likely that `ta`, `tai`,
and `taf` will have less values than when `ending` was set to False.

References
----------
.. [1] Gustafsson (2000) Adaptive Filtering and Change Detection.
.. [2] hhttp://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb

"""

from __future__ import division, print_function
import numpy as np
import pandas as pd

FILE_PATH = 'train_brent_spot.csv'
COLUMN = 'Dollars/Barrel'
THRESHOLD = 30
DRIFT = 1

def detect_cusum(x, threshold=1, drift=0, show=True, ax=None):

    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai = np.array([[], []], dtype=int)
    tap, tan = 0, 0
    
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    
    if show:
        _plot(x, threshold, drift, ax, ta, tai, gp, gn)

    return ta, tai


def _plot(x, threshold, drift, ax, ta, tai, gp, gn):
    """Plot results of the detect_cusum function, see its help."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(ta):
            ax1.plot(tai, x[tai], '>', mfc='g', mec='g', ms=10,
                     label='Start')
            
            ax1.plot(ta, x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                     label='Alarm')
            ax1.legend(loc='best', framealpha=.5, numpoints=1)
        
        file_path = FILE_PATH
        df = pd.read_csv(file_path)
        cps = df[df['label'] != 0].index.tolist()
        for i, cp in enumerate(cps):
            if i == 0:
                ax1.axvline(cp, c='red', ls='dotted', label='Actual CP')
            else:
                ax1.axvline(cp, c='red', ls='dotted')
            
        
        ax1.set_xlim(-.01*x.size, x.size*1.01-1)
        ax1.set_xlabel('Data #', fontsize=14)
        ax1.set_ylabel('Amplitude', fontsize=14)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax1.set_title('Time series and detected changes ' +
                      '(threshold= %.3g, drift= %.3g): N changes = %d'
                      % (threshold, drift, len(tai)))
        
        ax1.legend(loc='best')
        
        ax2.plot(t, gp, 'y-', label='+')
        ax2.plot(t, gn, 'm-', label='-')
        ax1.set_ylabel('Amplitude', fontsize=14)
        ax2.set_title('Time series of the cumulative sums of ' +
                      'positive and negative changes')
        ax2.legend(loc='best', framealpha=.5, numpoints=1)
        plt.tight_layout()
        plt.show()


file_path = FILE_PATH
df = pd.read_csv(file_path)
x = df[COLUMN].tolist()
detect_cusum(x, THRESHOLD, DRIFT, True)

