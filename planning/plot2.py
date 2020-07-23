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

import sequential_planning

def get_state(X, t):
  if t < X.size(0):
    return X[t]
  else:
    return X[-1]

def plot(ax1, ax2, robots, field, field2 = None):
  T = 0
  for robot in robots:
    X = getattr(robot, 'X_' + field)
    T = max(T, X.size(0))

  for k, robot in enumerate(robots):
    X = getattr(robot, 'X_' + field)
    U = getattr(robot, 'U_' + field)

    # plot trajectory
    line = ax1.plot(X[:,0], X[:,1],alpha=0.5)
    color = line[0].get_color()
    
    # plot desired state (dashed)
    if field2 is not None:
      X_des = getattr(robot, 'X_' + field2)
      ax1.plot(X_des[:,0], X_des[:,1], alpha=0.5, color=color, linestyle="--")

    # plot velocity vectors:
    qX = []
    qY = []
    qU = []
    qV = []
    for k in np.arange(0,X.shape[0], int(0.5 / dt)):
      qX.append(X[k,0])
      qY.append(X[k,1])
      qU.append(X[k,2])
      qV.append(X[k,3])
    ax1.quiver(qX,qY,qU,qV,angles='xy', scale_units='xy',scale=5, color=color, width=0.01)

    # plot outline
    # ax.add_artist(mpatches.Circle(get_state(X,0)[0:2], robot.radius, color=color, alpha=0.2))
    ax1.add_artist(mpatches.Circle(get_state(X,int(T/2))[0:2], robot.radius, color=color, alpha=0.1))




    ax1.set_aspect('equal')
    # xlim = ax[0,c].get_xlim()
    # ylim = ax[0,c].get_ylim()
    ax1.set_xlim([-0.4,0.7])
    ax1.set_ylim([0.6,1.4])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')

    # plot acceleration
    ax2.plot([i * dt for i in range(X.size(0))], X[:,4], color)
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel(r"$f_a$ [g]")

def vis_pdf(robotsNN, robotsBaseline, dt, name='output.pdf'):
  pp = PdfPages(name)

  scale = 2.5
  fig, ax = plt.subplots(2, 3, figsize=(8*scale,2.4*scale),sharex='row', sharey='row', gridspec_kw={'height_ratios': [5,1]})

  ax[0,0].set_title('Planning with NN (Stage 1)')
  plot(ax[0,0], ax[1,0], robotsNN, "treesearch")

  ax[0,1].set_title('Planning with NN (Tracking)')
  plot(ax[0,1], ax[1,1], robotsNN, "rollout", "des")

  ax[0,2].set_title('Planning without NN (Tracking)')
  plot(ax[0,2], ax[1,2], robotsBaseline, "rollout", "des")

  fig.subplots_adjust(wspace=0.1, hspace=0.1)

  pp.savefig(fig)
  plt.close(fig)


  pp.close()
  subprocess.call(["pdfcrop", name, name])
  subprocess.call(["xdg-open", name])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("visNN", help="visualize a *.p file")
  parser.add_argument("visBaseline", help="visualize a *.p file")
  args = parser.parse_args()

  plt.rcParams.update({'font.size': 14})
  plt.rcParams['lines.linewidth'] = 4

  dt = 0.05
  robotsNN = pickle.load(open(args.visNN, "rb"))
  sequential_planning.tracking(robotsNN, dt)

  robotsBaseline = pickle.load(open(args.visBaseline, "rb"))
  sequential_planning.tracking(robotsBaseline, dt)


  # tracking(robots, dt)
  vis_pdf(robotsNN, robotsBaseline, dt, "plot2.pdf")
