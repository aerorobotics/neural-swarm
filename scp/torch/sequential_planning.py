from robots import RobotCrazyFlie2D
from sequential_tree_search import tree_search
import torch
import math
from sequential_scp import scp_min_xf, scp
import numpy as np
import pickle
import subprocess
import random

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def add_result_figs(robots, pp, field, name):
  fig, ax = plt.subplots()
  for k, robot in enumerate(robots):
    X = getattr(robot, 'X_' + field)
    ax.plot(X[:,0], X[:,1], label="cf{}".format(k))
  ax.legend()
  ax.set_title('{} - position'.format(name))
  pp.savefig(fig)
  plt.close(fig)

  fig, ax = plt.subplots(2, 1)
  for k, robot in enumerate(robots):
    X = getattr(robot, 'X_' + field)
    U = getattr(robot, 'U_' + field)
    line = ax[0].plot(torch.norm(X[:,2:4], dim=1), label="cf{}".format(k))
    ax[1].plot(torch.norm(U, dim=1), line[0].get_color())
  ax[0].legend()
  ax[0].set_title('{} - Velocity'.format(name))
  ax[1].legend()
  ax[1].set_title('{} - Acceleration'.format(name))
  pp.savefig(fig)
  plt.close(fig)


def vis_pdf(robots, name='output.pdf'):
  # scp_epoch = len(X)

  pp = PdfPages(name)

  add_result_figs(robots, pp, 'treesearch', 'Tree Search')
  add_result_figs(robots, pp, 'scpminxf', 'SCP (min xf)')
  add_result_figs(robots, pp, 'des', 'SCP (min u)')

  # roll-out
  fig, ax = plt.subplots(2, 1)
  for k, robot in enumerate(robots):
    line = ax[0].plot(robot.X_des[:,0], robot.X_des[:,1], linestyle='--', label="cf{} des pos".format(k))
    ax[0].plot(robot.X_rollout[:,0], robot.X_rollout[:,1], line[0].get_color(), label="cf{} tracking".format(k))
    ax[1].plot(robot.Fa_rollout, line[0].get_color(), label="cf{} Fa".format(k))
  ax[0].legend()
  ax[0].set_title('Position tracking')
  ax[1].legend()
  ax[1].set_title('Fa')
  pp.savefig(fig)
  plt.close(fig)

  fig, ax = plt.subplots()
  for k, robot in enumerate(robots):
    line = ax.plot(robot.X_des[:,2], robot.X_des[:,3], linestyle='--', label="cf{} des vel".format(k))
    ax.plot(robot.X_rollout[:,2], robot.X_rollout[:,3], line[0].get_color(), label="cf{} tracking".format(k))
  ax.legend()
  ax.set_title('Velocity tracking')
  pp.savefig(fig)
  plt.close(fig)

  fig, ax = plt.subplots(1, 2)
  for k, robot in enumerate(robots):
    ax[0].plot(robot.U_rollout[:,0], label="cf{} uy".format(k))
    ax[0].plot(robot.U_rollout[:,1], label="cf{} uz".format(k))
    ax[1].plot(torch.norm(robot.U_rollout, dim=1), label="cf{} thrust".format(k))
  ax[0].legend()  
  ax[1].axhline(y=robot.g*robot.thrust_to_weight, linestyle='--', label="limit")
  ax[1].legend()
  ax[1].set_title('Control and thrust')
  pp.savefig(fig)
  plt.close()

  pp.close()
  subprocess.call(["xdg-open", name])



def tracking(robots, dt, feedforward=True):
  # find total time for all robots to reach their goal
  T = 0
  for robot in robots:
    T = max(T, robot.X_des.size(0))

  # for each robot allocate the required memory
  for robot in robots:
    robot.X_rollout = torch.zeros((T, robot.stateDim))
    robot.U_rollout = torch.zeros((T-1, robot.ctrlDim))
    robot.Fa_rollout = torch.zeros((T-1,))
    robot.X_rollout[0] = robot.x0

  # simulate the execution
  for t in range(T-1):
    for k, robot in enumerate(robots):
      # compute neighboring states:
      x_neighbors = []
      for k_other, robot_other in enumerate(robots):
        if k != k_other:
          x_neighbors.append(robot_other.X_rollout[t])

      # control + forward propagation
      if feedforward and t < robot.U_des.size(0):
        v_d_dot = robot.U_des[t]
      else:
        # v_d_dot = torch.zeros(robot.ctrlDim)
        v_d_dot = torch.tensor([0, robot.g])

      if t < robot.X_des.size(0):
        x_d = robot.X_des[t]
      else:
        x_d = robot.X_des[-1]

      u = robot.controller(x=robot.X_rollout[t], x_d=x_d, v_d_dot=v_d_dot)
      robot.U_rollout[t] = u
      dx = robot.f(robot.X_rollout[t], robot.U_rollout[t], x_neighbors, useNN=True)
      robot.X_rollout[t+1] = robot.X_rollout[t] + dt*dx
      robot.Fa_rollout[t] = robot.compute_Fa(robot.X_rollout[t], x_neighbors, True)

  # TODO: disable grad by default...
  for robot in robots:
    robot.X_rollout = robot.X_rollout.detach()
    robot.U_rollout = robot.U_rollout.detach()
    robot.Fa_rollout = robot.Fa_rollout.detach()

  # energy = torch.sum(U.norm(dim=1) * dt).item()
  # tracking_error = torch.sum(torch.norm(X - X_d, dim=1) * dt).item()
  # print("energy: ", energy, " tracking error: ", tracking_error)
  # return X.detach(), U.detach(), Fa.detach()

if __name__ == '__main__':
  dt = 0.05
  robots = []
  useNN = True

  # CF0
  robot = RobotCrazyFlie2D(useNN=useNN)
  robot.x0 = torch.tensor([-0.5,0,0,0], dtype=torch.float32)
  robot.xf = torch.tensor([0.5,0,0,0], dtype=torch.float32)
  robots.append(robot)
  # robot = pickle.load( open( "robot0.p", "rb" ) )
  # robots.append(robot)

  # CF1
  robot = RobotCrazyFlie2D(useNN=useNN)
  robot.x0 = torch.tensor([0.5,0,0,0], dtype=torch.float32)
  robot.xf = torch.tensor([-0.5,0,0,0], dtype=torch.float32)
  robots.append(robot)
  # robot = pickle.load( open( "robot1.p", "rb" ) )
  # robots.append(robot)

  # # CF2
  # robot = RobotCrazyFlie2D(useNN=useNN)
  # robot.x0 = torch.tensor([-0.0,0,0,0], dtype=torch.float32)
  # robot.xf = torch.tensor([1.0,0,0,0], dtype=torch.float32)
  # robots.append(robot)

  # # Variant 1: fully sequential

  # for k, robot in enumerate(robots):
  #   other_x = [r.X_des for r in robots[0:k]]

  #   # run tree search to find initial solution
  #   x, u = tree_search(robot, robot.x0, robot.xf, dt, other_x, prop_iter=2, iters=20000, top_k=100, trials=3)
  #   robot.X_treesearch = x
  #   robot.U_treesearch = u

  #   # run scp (min xf) to exactly go to the goal
  #   X1, U1, X1_integration = scp_min_xf(robot, robot.X_treesearch, robot.U_treesearch, robot.xf, dt, other_x, trust_region=True, trust_x=0.25, trust_u=1, num_iterations=1000)
  #   robot.X_scpminxf = X1[-1]
  #   robot.U_scpminxf = U1[-1]

  #   # run scp to minimize U
  #   X2, U2, X2_integration = scp(robot, X1[-1], U1[-1], dt, other_x, trust_region=True, trust_x=0.25, trust_u=1, num_iterations=10)
  #   robot.X_des = X2[-1]
  #   robot.U_des = U2[-1]

  # Variant 2: sequential tree search, followed by iterative SCP

  for k, robot in enumerate(robots):
    other_x = [r.X_treesearch for r in robots[0:k]]

    # run tree search to find initial solution
    x, u = tree_search(robot, robot.x0, robot.xf, dt, other_x, prop_iter=2, iters=20000, top_k=100, trials=3)
    robot.X_treesearch = x
    robot.U_treesearch = u
    # setup for later iterative refinement
    robot.X_scpminxf = x
    robot.U_scpminxf = u

  permutation = list(range(len(robots)))

  # run scp (min xf) to exactly go to the goal
  for iteration in range(10):
    random.shuffle(permutation)
    for idx in permutation:
      robot = robots[idx]
      other_x = [r.X_scpminxf for r in robots[0:idx] + robots[idx+1:]]
      X1, U1, X1_integration = scp_min_xf(robot, robot.X_scpminxf, robot.U_scpminxf, robot.xf, dt, other_x, trust_region=True, trust_x=0.25, trust_u=1, num_iterations=1)
      robot.X_scpminxf = X1[-1]
      robot.U_scpminxf = U1[-1]
      # setup for next stage
      robot.X_des = robot.X_scpminxf
      robot.U_des = robot.U_scpminxf

  # run scp to minimize U
  for iteration in range(10):
    random.shuffle(permutation)
    for idx in permutation:
      robot = robots[idx]
      other_x = [r.X_scpminxf for r in robots[0:idx] + robots[idx+1:]]

      X2, U2, X2_integration = scp(robot, robot.X_des, robot.U_des, dt, other_x, trust_region=True, trust_x=0.25, trust_u=1, num_iterations=1)
      robot.X_des = X2[-1]
      robot.U_des = U2[-1]

  # pickle.dump( robots[1], open( "robot1.p", "wb" ) )

  tracking(robots, dt)
  vis_pdf(robots, "output.pdf")
