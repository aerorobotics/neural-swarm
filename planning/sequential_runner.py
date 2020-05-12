import concurrent.futures
from itertools import repeat
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sequential_planning import sequential_planning

if __name__ == '__main__':

  jobs = 5

  if True:
    useNNs = []
    file_names = []
    for useNN in [True, False]:
      for j in range(jobs):
        useNNs.append(useNN)
        if useNN:
          file_names.append("output{}NN.pdf".format(j))
        else:
          file_names.append("output{}.pdf".format(j))

    with concurrent.futures.ProcessPoolExecutor() as executor:
      results = executor.map(sequential_planning, useNNs, file_names)
      results = list(results)

    pickle.dump( results, open( "runner.p", "wb" ) )
  else:
    results = pickle.load( open( "runner.p", "rb" ) )

  # compute position tracking error and control effort statistics
  tracking_errors = np.zeros(len(results))
  control_efforts = np.zeros(len(results))
  for k, stats in enumerate(results):
    tracking_errors[k] = stats['tracking_errors'][0]
    control_efforts[k] = stats['control_efforts'][0]

  fig, ax = plt.subplots(1, 2)
  ax[0].set_title('Tracking errors')
  ax[0].boxplot([tracking_errors[0:jobs], tracking_errors[jobs:]], labels=["NN", "no NN"])

  ax[1].set_title('Control efforts')
  ax[1].boxplot([control_efforts[0:jobs], control_efforts[jobs:]], labels=["NN", "no NN"])

  plt.show()
