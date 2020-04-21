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
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

def get_state(X, t):
  if t < X.size(0):
    return X[t]
  else:
    return X[-1]

def add_result_figs(robots, pp, field, name, animate = False):
  # position
  fig, ax = plt.subplots()
  for k, robot in enumerate(robots):
    X = getattr(robot, 'X_' + field)
    ax.plot(X[:,0], X[:,1], label="cf{}".format(k))
  ax.set_aspect('equal')
  ax.legend()
  ax.set_title('{} - position'.format(name))
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  pp.savefig(fig)
  plt.close(fig)

  # min distance between any two robots
  T = 0
  for robot in robots:
    X = getattr(robot, 'X_' + field)
    T = max(T, X.size(0))

  dist = torch.ones((T,)) * 10
  for i in range(len(robots)):
    X1 = getattr(robots[i], 'X_' + field)
    X1 = torch.stack([get_state(X1, t) for t in range(T)])
    for j in range(i+1,len(robots)):
      X2 = getattr(robots[j], 'X_' + field)
      X2 = torch.stack([get_state(X2, t) for t in range(T)])
      dist_12 = torch.norm(X1[:,0:2] - X2[:,0:2], dim=1)
      dist = torch.min(dist, dist_12)

  fig, ax = plt.subplots()
  ax.plot(dist)
  ax.axhline(y=2*robots[0].radius, linestyle='--', label="limit")
  ax.set_title('{} - min distance'.format(name))
  pp.savefig(fig)
  plt.close(fig)

  # velocity
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

  if "SCP" in name:
    fig, ax = plt.subplots()
    for k, robot in enumerate(robots):
      obj_values = getattr(robot, 'obj_values_' + field)
      ax.plot(obj_values, label="cf{}".format(k))
    ax.legend()
    ax.set_title('{} - objective values'.format(name))
    pp.savefig(fig)
    plt.close(fig)

  if animate:
    for t in range(T):
      fig, ax = plt.subplots()
      ax.set_aspect('equal')
      ax.set_xlim(xlim + np.array([-0.2,0.2]))
      ax.set_ylim(ylim + np.array([-0.2,0.2]))
      for k, robot in enumerate(robots):
        X = getattr(robot, 'X_' + field)
        ax.add_artist(mpatches.Circle(get_state(X,t)[0:2], robot.radius))
      ax.set_title('{} - t = {}'.format(name, t))
      pp.savefig(fig)
      plt.close(fig)

def vis_pdf(robots, name='output.pdf'):
  # scp_epoch = len(X)

  pp = PdfPages(name)

  add_result_figs(robots, pp, 'treesearch', 'Tree Search')
  add_result_figs(robots, pp, 'scpminxf', 'SCP (min xf)')
  add_result_figs(robots, pp, 'des', 'SCP (min u)', animate=True)

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
  dt = 0.025
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

  # CF2
  robot = RobotCrazyFlie2D(useNN=useNN)
  robot.x0 = torch.tensor([-0.0,0,0,0], dtype=torch.float32)
  robot.xf = torch.tensor([1.0,0,0,0], dtype=torch.float32)
  robots.append(robot)

  # CF3
  robot = RobotCrazyFlie2D(useNN=useNN)
  robot.x0 = torch.tensor([1.0,0,0,0], dtype=torch.float32)
  robot.xf = torch.tensor([0.0,0,0,0], dtype=torch.float32)
  robots.append(robot)

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

  if True:

    for k, robot in enumerate(robots):
      print("Tree search for robot {}".format(k))
      other_x = [r.X_treesearch for r in robots[0:k]]

      # run tree search to find initial solution
      x, u = tree_search(robot, robot.x0, robot.xf, dt, other_x, prop_iter=4, iters=10000, top_k=100, trials=3)
      robot.X_treesearch = x
      robot.U_treesearch = u
      # setup for later iterative refinement
      robot.X_scpminxf = x
      robot.U_scpminxf = u
      robot.obj_values_scpminxf = []

    pickle.dump( robots, open( "robots.p", "wb" ) )
  else:
    robots = pickle.load( open( "robots.p", "rb" ) )

  # vis_pdf(robots, "output.pdf")
  # exit()


  permutation = list(range(len(robots)))

  # run scp (min xf) to exactly go to the goal
  for iteration in range(100):
    random.shuffle(permutation)
    all_robots_done = True
    for idx in permutation:
      print("SCP (min xf) for robot {}".format(idx))
      robot = robots[idx]
      other_x = [r.X_scpminxf for r in robots[0:idx] + robots[idx+1:]]
      X1, U1, X1_integration, obj_value = scp_min_xf(robot, robot.X_scpminxf, robot.U_scpminxf, robot.xf, dt, other_x, trust_region=True, trust_x=0.25, trust_u=1, num_iterations=1)
      robot.X_scpminxf = X1[-1]
      robot.U_scpminxf = U1[-1]
      robot.obj_values_scpminxf.append(obj_value)
      print("... finished with obj value {}".format(obj_value))
      # setup for next stage
      robot.X_des = robot.X_scpminxf
      robot.U_des = robot.U_scpminxf
      robot.obj_values_des = []
      if obj_value > 1e-8:
        all_robots_done = False
    if all_robots_done:
      break

  # run scp to minimize U
  for iteration in range(20):
    random.shuffle(permutation)
    for idx in permutation:
      print("SCP (min u) for robot {}".format(idx))
      robot = robots[idx]
      other_x = [r.X_des for r in robots[0:idx] + robots[idx+1:]]

      X2, U2, X2_integration, obj_value = scp(robot, robot.X_des, robot.U_des, dt, other_x, trust_region=True, trust_x=0.25, trust_u=1, num_iterations=1)
      robot.X_des = X2[-1]
      robot.U_des = U2[-1]
      robot.obj_values_des.append(obj_value)
      print("... finished with obj value {}".format(obj_value))

  # pickle.dump( robots[1], open( "robot1.p", "wb" ) )

  tracking(robots, dt)
  vis_pdf(robots, "output.pdf")
