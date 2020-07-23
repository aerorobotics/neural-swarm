import concurrent.futures
from itertools import repeat
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

import sequential_planning

if __name__ == '__main__':

  plt.rcParams.update({'font.size': 12})
  plt.rcParams['lines.linewidth'] = 2

  repeats = 5
  cases = ["SL", "SS", "LL", "SSL"]
  dt = 0.05

  results = []
  for case in cases:
    for useNN in [True, False]:
      for j in range(repeats):
        if useNN:
          file_name = "../data/planning/case_{}_iter_{}_NN.p".format(case, j)
        else:
          file_name = "../data/planning/case_{}_iter_{}.p".format(case, j)
        print(file_name)
        robots = pickle.load(open(file_name, "rb"))
        sequential_planning.tracking(robots, dt)
        stats = sequential_planning.compute_stats(robots, dt)
        results.append(stats)

  # compute position tracking error and control effort statistics
  max_z_errors = np.zeros(len(results))
  fa_abs_max_small = np.zeros(len(results))
  fa_abs_max_large = np.zeros(len(results))
  for k, stats in enumerate(results):
    max_z_errors[k] = stats['max_z_errors'][0]
    if 'small' in stats['fa_abs_max']:
      fa_abs_max_small[k] = stats['fa_abs_max']['small']
    if 'large' in stats['fa_abs_max']:
      fa_abs_max_large[k] = stats['fa_abs_max']['large']

  boxprops = dict()
  medianprops = dict(linewidth=3, color='forestgreen')

  fig, ax = plt.subplots(2, len(cases), sharey='row', squeeze=False)
  for c, case in enumerate(cases):
    idx = c * 2 * repeats

    ax[0,c].set_title(case)

    ax[0,c].boxplot([max_z_errors[idx:idx+repeats], max_z_errors[idx+repeats:idx+2*repeats]], 
      widths=0.75, showfliers=False, boxprops=boxprops, medianprops=medianprops, 
      labels=["NN", "no NN"])
    ax[0,c].axvspan(0.5, 1.5, facecolor='black', alpha=0.2)

    # ax[1,c].set_title('Control efforts ' + case['name'])
    ax[1,c].boxplot([
      fa_abs_max_small[idx:idx+repeats],
      fa_abs_max_small[idx+repeats:idx+2*repeats],
      fa_abs_max_large[idx:idx+repeats],
      fa_abs_max_large[idx+repeats:idx+2*repeats]],
      widths=0.75, showfliers=False, boxprops=boxprops, medianprops=medianprops, 
      labels=[r"$S_{NN}$", r"$S_{BL}$", r"$L_{NN}$", r"$L_{BL}$"])
    ax[1,c].axhline(y=5, xmin=0, xmax=0.5, linestyle='--', color='firebrick')
    ax[1,c].axhline(y=10, xmin=0.5, xmax=1.0, linestyle='--', color='firebrick')

    ax[0,c].grid(True, axis='y')
    ax[1,c].grid(True, axis='y')
    ax[1,c].axvspan(0.5, 1.5, facecolor='black', alpha=0.2)
    ax[1,c].axvspan(2.5, 3.5, facecolor='black', alpha=0.2)


    # ax[2,c].bar(0, np.sum(fa_abs_max[idx:idx+repeats]))#, labels=["NN", "no NN"])
    # ax[2,c].bar(1, np.sum(fa_abs_max[idx+repeats:idx+2*repeats]))#, labels=["NN", "no NN"])

  # ax[0,0].set_ylabel('Tracking error')
  ax[0,0].set_ylabel('Max z error [m]')
  ax[1,0].set_ylabel(r"$\max |f_a|$ [g]")
  # ax[3,0].set_ylabel('# Success')

  plt.show()
