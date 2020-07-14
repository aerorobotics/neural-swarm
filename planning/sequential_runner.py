import concurrent.futures
from itertools import repeat
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from sequential_planning import sequential_planning

if __name__ == '__main__':

  plt.rcParams.update({'font.size': 12})
  plt.rcParams['lines.linewidth'] = 4

  max_workers = 24

  repeats = 5

  cases = [
    {
      'name': "Small/Large Swap",
      'shortname': "SL",
      'use3D': False,
      'robots': [
        {
          'type': 'small',
          'x0': torch.tensor([-0.25,1,0,0,0], dtype=torch.float32),
          'xf': torch.tensor([0.25,1,0,0,0], dtype=torch.float32),
        },
        {
          'type': 'large',
          'x0': torch.tensor([0.25,1,0,0,0], dtype=torch.float32),
          'xf': torch.tensor([-0.25,1,0,0,0], dtype=torch.float32),
        },
      ]
    },
    {
      'name': "Small/Small Swap",
      'shortname': "SS",
      'use3D': False,
      'robots': [
        {
          'type': 'small',
          'x0': torch.tensor([-0.25,1,0,0,0], dtype=torch.float32),
          'xf': torch.tensor([0.25,1,0,0,0], dtype=torch.float32),
        },
        {
          'type': 'small',
          'x0': torch.tensor([0.25,1,0,0,0], dtype=torch.float32),
          'xf': torch.tensor([-0.25,1,0,0,0], dtype=torch.float32),
        },
      ]
    },
    {
      'name': "Large/Large Swap",
      'shortname': "LL",
      'use3D': False,
      'robots': [
        {
          'type': 'large',
          'x0': torch.tensor([-0.25,1,0,0,0], dtype=torch.float32),
          'xf': torch.tensor([0.25,1,0,0,0], dtype=torch.float32),
        },
        {
          'type': 'large',
          'x0': torch.tensor([0.25,1,0,0,0], dtype=torch.float32),
          'xf': torch.tensor([-0.25,1,0,0,0], dtype=torch.float32),
        },
      ]
    },
    # {
    #   'name': "Small/Large Swap (3D)",
    #   'shortname': "SL3D",
    #   'use3D': True,
    #   'robots': [
    #     {
    #       'type': 'small',
    #       'x0': torch.tensor([0,-0.5,0,0,0,0], dtype=torch.float32),
    #       'xf': torch.tensor([0,0.5,0,0,0,0], dtype=torch.float32),
    #     },
    #     {
    #       'type': 'large',
    #       'x0': torch.tensor([0,0.5,0,0,0,0], dtype=torch.float32),
    #       'xf': torch.tensor([0,-0.5,0,0,0,0], dtype=torch.float32),
    #     },
    #   ]
    # },
  ]

  if False:
    useNNs = []
    robot_infos = []
    file_names = []
    for case in cases:
      for useNN in [True, False]:
        for j in range(repeats):
          useNNs.append(useNN)
          robot_infos.append(case)
          if useNN:
            file_names.append("../data/planning/case_{}_iter_{}_NN.pdf".format(case['shortname'], j))
          else:
            file_names.append("../data/planning/case_{}_iter_{}.pdf".format(case['shortname'], j))

    with concurrent.futures.ProcessPoolExecutor(max_workers) as executor:
      results = executor.map(sequential_planning, 
        useNNs,
        [ri['robots'] for ri in robot_infos],
        file_names,
        [ri['use3D'] for ri in robot_infos])#,
        # repeat(200))
      results = list(results)

    pickle.dump( results, open( "runner.p", "wb" ) )
  else:
    results = pickle.load( open( "runner.p", "rb" ) )

  # compute position tracking error and control effort statistics
  tracking_errors = np.zeros(len(results))
  max_z_errors = np.zeros(len(results))
  control_efforts = np.zeros(len(results))
  fa_within_bounds = np.zeros(len(results))
  for k, stats in enumerate(results):
    tracking_errors[k] = stats['tracking_errors'][0]
    max_z_errors[k] = stats['max_z_errors'][0]
    # control_efforts[k] = stats['control_efforts'][0]
    # tracking_errors[k] = stats['tracking_errors_avg']
    control_efforts[k] = stats['control_efforts_avg']
    fa_within_bounds[k] = stats['fa_within_bounds']

  fig, ax = plt.subplots(4, len(cases), sharey='row', squeeze=False)
  for c, case in enumerate(cases):
    idx = c * 2 * repeats

    ax[0,c].set_title(case['name'])
    ax[0,c].boxplot([tracking_errors[idx:idx+repeats], tracking_errors[idx+repeats:idx+2*repeats]], labels=["NN", "no NN"])

    ax[1,c].boxplot([max_z_errors[idx:idx+repeats], max_z_errors[idx+repeats:idx+2*repeats]], labels=["NN", "no NN"])

    # ax[1,c].set_title('Control efforts ' + case['name'])
    ax[2,c].boxplot([control_efforts[idx:idx+repeats], control_efforts[idx+repeats:idx+2*repeats]], labels=["NN", "no NN"])

    ax[0,c].grid(True)
    ax[1,c].grid(True)
    ax[2,c].grid(True)

    ax[3,c].bar(0, np.sum(fa_within_bounds[idx:idx+repeats]))#, labels=["NN", "no NN"])
    ax[3,c].bar(1, np.sum(fa_within_bounds[idx+repeats:idx+2*repeats]))#, labels=["NN", "no NN"])

  ax[0,0].set_ylabel('Tracking error')
  ax[1,0].set_ylabel('Max z error')
  ax[2,0].set_ylabel('Avg. control effort')
  ax[3,0].set_ylabel('# Success')

  plt.show()
