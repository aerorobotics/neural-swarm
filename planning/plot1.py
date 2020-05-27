import torch
import math
import numpy as np
import pickle
import subprocess
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

def get_state(X, t):
  if t < X.size(0):
    return X[t]
  else:
    return X[-1]

def vis_pdf(robots, dt, name='output.pdf'):
  pp = PdfPages(name)

  scale = 2.5
  fig, ax = plt.subplots(2, 3, figsize=(8*scale,2.0*scale),sharex='row', sharey='row', gridspec_kw={'height_ratios': [5,1]})

  for c, field in enumerate(["treesearch", "scpminxf", "des"]):

    T = 0
    for robot in robots:
      X = getattr(robot, 'X_' + field)
      T = max(T, X.size(0))

    for k, robot in enumerate(robots):
      X = getattr(robot, 'X_' + field)
      U = getattr(robot, 'U_' + field)

      # plot trajectory
      line = ax[0,c].plot(X[:,0], X[:,1],alpha=0.5)
      color = line[0].get_color()

      # plot velocity vectors:
      qX = []
      qY = []
      qU = []
      qV = []
      for k in np.arange(0,X.shape[0], 2):#int(5.0 / dt)):
        qX.append(X[k,0])
        qY.append(X[k,1])
        qU.append(X[k,2])
        qV.append(X[k,3])
      ax[0,c].quiver(qX,qY,qU,qV,angles='xy', scale_units='xy',scale=25, color=color, width=0.01)

      # plot outline
      # ax.add_artist(mpatches.Circle(get_state(X,0)[0:2], robot.radius, color=color, alpha=0.2))
      ax[0,c].add_artist(mpatches.Circle(get_state(X,int(T/2))[0:2], robot.radius, color=color, alpha=0.1))


      ax[0,c].set_aspect('equal')
      # xlim = ax[0,c].get_xlim()
      # ylim = ax[0,c].get_ylim()
      ax[0,c].set_xlim([-0.7,0.7])
      ax[0,c].set_ylim([-0.55,0.25])
      ax[0,c].set_xticklabels([])
      ax[0,c].set_yticklabels([])

      # plot acceleration
      ax[1,c].plot([i * dt for i in range(U.size(0))], torch.norm(U, dim=1), color)

    # ax.set_xlabel("Y [m]")
    # ax.set_ylabel("Z [m]")
  fig.subplots_adjust(wspace=0.1, hspace=0.0)
  ax[0,0].set_title('Stage 1')
  ax[0,1].set_title('Stage 2')
  ax[0,2].set_title('Stage 3')
  pp.savefig(fig)
  plt.close(fig)


  pp.close()
  subprocess.call(["xdg-open", name])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--vis", help="visualize a *.p file")
  args = parser.parse_args()

  plt.rcParams.update({'font.size': 14})
  plt.rcParams['lines.linewidth'] = 4

  dt = 0.05
  robots = pickle.load(open(args.vis, "rb"))
  # tracking(robots, dt)
  vis_pdf(robots, dt, "plot1.pdf")
