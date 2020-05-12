from robots import RobotCrazyFlie2D, RobotCrazyFlie3D
# from sequential_tree_search import tree_search
from sequential_tree_search_ao_rrt import tree_search
import torch
import math
from sequential_scp import scp_min_xf, scp
import numpy as np
import pickle
import subprocess
import random
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

# Helper functions to have 3D plot with equal axis, 
# see: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def add_result_figs(robots, pp, field, name, use3D=False, animate = False):
  # position
  if use3D:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for k, robot in enumerate(robots):
      X = np.array(getattr(robot, 'X_' + field))
      ax.plot3D(X[:,0], X[:,1], X[:,2], label="cf{}".format(k))
    set_axes_equal(ax)
    ax.legend()
    ax.set_title('{} - position'.format(name))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    pp.savefig(fig)
    plt.close(fig)
  else:
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
      if use3D:
        dist_12 = torch.norm(X1[:,0:3] - X2[:,0:3], dim=1)
      else:
        dist_12 = torch.norm(X1[:,0:2] - X2[:,0:2], dim=1)
      dist_between_shapes = dist_12 - (robots[i].radius + robots[j].radius)
      dist = torch.min(dist, dist_between_shapes)

  fig, ax = plt.subplots()
  ax.plot(dist)
  ax.axhline(y=0, linestyle='--', label="limit")
  ax.set_title('{} - shape min distance'.format(name))
  pp.savefig(fig)
  plt.close(fig)

  # velocity
  colors = []
  fig, ax = plt.subplots(2, 1)
  for k, robot in enumerate(robots):
    X = getattr(robot, 'X_' + field)
    U = getattr(robot, 'U_' + field)
    if use3D:
      line = ax[0].plot(torch.norm(X[:,3:6], dim=1), label="cf{}".format(k))
    else:
      line = ax[0].plot(torch.norm(X[:,2:4], dim=1), label="cf{}".format(k))
    ax[1].plot(torch.norm(U, dim=1), line[0].get_color())
    colors.append(line[0].get_color())
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
      if use3D:

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(xlim + np.array([-0.2,0.2]))
        ax.set_ylim(ylim + np.array([-0.2,0.2]))
        ax.set_zlim(zlim + np.array([-0.2,0.2]))
        set_axes_equal(ax)
        for k, robot in enumerate(robots):
          X = getattr(robot, 'X_' + field)
          Xt = np.array(get_state(X,t))

          u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
          x = robot.radius * np.cos(u)*np.sin(v) + Xt[0]
          y = robot.radius * np.sin(u)*np.sin(v) + Xt[1]
          z = robot.radius * np.cos(v) + Xt[2]
          ax.plot_wireframe(x, y, z, color=colors[k])
        ax.set_title('{} - t = {}'.format(name, t))
        pp.savefig(fig)
        plt.close(fig)

      else:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(xlim + np.array([-0.2,0.2]))
        ax.set_ylim(ylim + np.array([-0.2,0.2]))
        for k, robot in enumerate(robots):
          X = getattr(robot, 'X_' + field)
          ax.add_artist(mpatches.Circle(get_state(X,t)[0:2], robot.radius, color=colors[k]))
        ax.set_title('{} - t = {}'.format(name, t))
        pp.savefig(fig)
        plt.close(fig)

def compute_stats(robots, dt):
  stats = dict()

  # compute position tracking error and control effort statistics
  tracking_errors = [0]
  control_efforts = [0]
  for k, robot in enumerate(robots):
    velIdx = robot.X_rollout.shape[1] // 2
    tracking_error = np.sum(np.linalg.norm(robot.X_rollout[0:robot.X_des.shape[0],0:velIdx] - robot.X_des[:,0:velIdx], axis=1))
    tracking_errors.append(tracking_error)
    tracking_errors[0] += tracking_error

    control_effort = np.sum(np.linalg.norm(robot.U_rollout, axis=1)) * dt
    control_efforts.append(control_effort)
    control_efforts[0] += control_effort

  stats['tracking_errors'] = tracking_errors
  stats['control_efforts'] = control_efforts

  return stats

def vis_pdf(robots, stats, name='output.pdf', use3D=False):
  # scp_epoch = len(X)

  pp = PdfPages(name)

  add_result_figs(robots, pp, 'treesearch', 'Tree Search',use3D=use3D)
  add_result_figs(robots, pp, 'scpminxf', 'SCP (min xf)',use3D=use3D)
  add_result_figs(robots, pp, 'des', 'SCP (min u)', use3D=use3D, animate=True)

  # roll-out
  if use3D:
    fig = plt.figure()
    ax0 = fig.add_subplot(2,1,1,projection='3d')
    ax1 = fig.add_subplot(2,1,2)
    for k, robot in enumerate(robots):
      X_des = np.array(robot.X_des)
      X_rollout = np.array(robot.X_rollout)
      Fa_rollout = np.array(robot.Fa_rollout)
      line = ax0.plot3D(X_des[:,0], X_des[:,1], X_des[:,2], linestyle='--', label="cf{} des pos".format(k))
      ax0.plot(X_rollout[:,0], X_rollout[:,1], X_rollout[:,2], line[0].get_color(), label="cf{} tracking".format(k))
      ax1.plot(Fa_rollout, line[0].get_color(), label="cf{} Fa".format(k))
    set_axes_equal(ax0)
    ax0.legend()
    ax0.set_title('Position tracking')
    ax1.legend()
    ax1.set_title('Fa')
    pp.savefig(fig)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for k, robot in enumerate(robots):
      X_des = np.array(robot.X_des)
      X_rollout = np.array(robot.X_rollout)
      line = ax.plot3D(X_des[:,3], X_des[:,4], X_des[:,5], linestyle='--', label="cf{} des vel".format(k))
      ax.plot(X_rollout[:,3], X_rollout[:,4], X_rollout[:,5], line[0].get_color(), label="cf{} tracking".format(k))
    set_axes_equal(ax)
    ax.legend()
    ax.set_title('Velocity tracking')
    pp.savefig(fig)
    plt.close(fig)

  else:
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
    if use3D:
      ax[0].plot(robot.U_rollout[:,0], label="cf{} ux".format(k))
      ax[0].plot(robot.U_rollout[:,1], label="cf{} uy".format(k))
      ax[0].plot(robot.U_rollout[:,2], label="cf{} uz".format(k))
    else:
      ax[0].plot(robot.U_rollout[:,0], label="cf{} uy".format(k))
      ax[0].plot(robot.U_rollout[:,1], label="cf{} uz".format(k))
    line = ax[1].plot(torch.norm(robot.U_rollout, dim=1), label="cf{} thrust".format(k))
    ax[1].axhline(y=robot.g*robot.thrust_to_weight, color=line[0].get_color(), linestyle='--')
  ax[0].legend()  
  ax[1].legend()
  ax[1].set_title('Control and thrust')
  pp.savefig(fig)
  plt.close()

  tracking_errors = stats['tracking_errors']
  control_efforts = stats['control_efforts']

  fig, ax = plt.subplots(1, 2)
  ax[0].set_title('Tracking errors')
  ax[0].bar(range(len(tracking_errors)), tracking_errors)

  ax[1].set_title('Control efforts')
  ax[1].bar(range(len(control_efforts)), control_efforts)

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
      data_neighbors = []
      for k_other, robot_other in enumerate(robots):
        if k != k_other:
          data_neighbors.append((robot_other.cftype, robot_other.X_rollout[t]))

      # control + forward propagation
      if feedforward and t < robot.U_des.size(0):
        v_d_dot = robot.U_des[t]
      else:
        v_d_dot = torch.zeros(robot.ctrlDim)
        v_d_dot[-1] = robot.g

      if t < robot.X_des.size(0):
        x_d = robot.X_des[t]
      else:
        x_d = robot.X_des[-1]

      u = robot.controller(x=robot.X_rollout[t], x_d=x_d, v_d_dot=v_d_dot)
      robot.U_rollout[t] = u
      dx = robot.f(robot.X_rollout[t], robot.U_rollout[t], data_neighbors, useNN=True)
      robot.X_rollout[t+1] = robot.X_rollout[t] + dt*dx
      robot.Fa_rollout[t] = robot.compute_Fa(robot.X_rollout[t], data_neighbors, True)

  # TODO: disable grad by default...
  for robot in robots:
    robot.X_rollout = robot.X_rollout.detach()
    robot.U_rollout = robot.U_rollout.detach()
    robot.Fa_rollout = robot.Fa_rollout.detach()

  # energy = torch.sum(U.norm(dim=1) * dt).item()
  # tracking_error = torch.sum(torch.norm(X - X_d, dim=1) * dt).item()
  # print("energy: ", energy, " tracking error: ", tracking_error)
  # return X.detach(), U.detach(), Fa.detach()

def sequential_planning(useNN, file_name="output.pdf", use3D=False):
  dt = 0.05
  robots = []
  model_folder = '../data/models/models_19and20/epoch5_lip3_5'

  if use3D:
    # CF0
    robot = RobotCrazyFlie3D(model_folder, useNN=useNN, cftype="small")
    robot.x0 = torch.tensor([0,-0.5,0,0,0,0], dtype=torch.float32)
    robot.xf = torch.tensor([0,0.5,0,0,0,0], dtype=torch.float32)
    robots.append(robot)

    # CF1
    robot = RobotCrazyFlie3D(model_folder, useNN=useNN, cftype="large")
    robot.x0 = torch.tensor([0,0.5,0,0,0,0], dtype=torch.float32)
    robot.xf = torch.tensor([0,-0.5,0,0,0,0], dtype=torch.float32)
    robots.append(robot)

  else:
    # 2D version

    # CF0
    robot = RobotCrazyFlie2D(model_folder, useNN=useNN, cftype="small")
    robot.x0 = torch.tensor([-0.5,0,0,0], dtype=torch.float32)
    robot.xf = torch.tensor([0.5,0,0,0], dtype=torch.float32)
    robots.append(robot)

    # CF1
    robot = RobotCrazyFlie2D(model_folder, useNN=useNN, cftype="large")
    robot.x0 = torch.tensor([0.5,0,0,0], dtype=torch.float32)
    robot.xf = torch.tensor([-0.5,0,0,0], dtype=torch.float32)
    robots.append(robot)

    # # CF2
    # robot = RobotCrazyFlie2D(useNN=useNN)
    # robot.x0 = torch.tensor([-0.0,0,0,0], dtype=torch.float32)
    # robot.xf = torch.tensor([1.0,0,0,0], dtype=torch.float32)
    # robots.append(robot)

    # # CF3
    # robot = RobotCrazyFlie2D(useNN=useNN)
    # robot.x0 = torch.tensor([1.0,0,0,0], dtype=torch.float32)
    # robot.xf = torch.tensor([0.0,0,0,0], dtype=torch.float32)
    # robots.append(robot)


  # tree search to find initial solution
  permutation = list(range(len(robots)))

  if True:
    cost_limits = np.repeat(1e6, len(robots))

    while (cost_limits == 1e6).any():
      for trial in range(5):
        print("Tree search trial {}".format(trial))
        random.shuffle(permutation)
        for idx in permutation:
          print("Tree search for robot {}".format(idx))
          robot = robots[idx]
          data_neighbors = [(r.cftype, r.X_treesearch) for r in robots[0:idx] + robots[idx+1:] if hasattr(r, 'X_treesearch')]

          # run tree search to find initial solution
          x, u, best_cost = tree_search(robot, robot.x0, robot.xf, dt, data_neighbors, prop_iter=2, iters=50000, top_k=100, trials=1, cost_limit=cost_limits[idx])
          if x is not None:
            cost_limits[idx] = 0.9 * best_cost
            robot.X_treesearch = x
            robot.U_treesearch = u
            # setup for later iterative refinement
            robot.X_scpminxf = x
            robot.U_scpminxf = u
            robot.obj_values_scpminxf = []

    pickle.dump( robots, open( "robots.p", "wb" ) )
  else:
    robots = pickle.load( open( "robots.p", "rb" ) )

  # vis_pdf(robots, "output.pdf", use3D=use3D)
  # exit()




  # run scp (min xf) to exactly go to the goal
  for iteration in range(100):
    random.shuffle(permutation)
    all_robots_done = True
    for idx in permutation:
      print("SCP (min xf) for robot {}".format(idx))
      robot = robots[idx]
      data_neighbors = [(r.cftype, r.X_scpminxf) for r in robots[0:idx] + robots[idx+1:]]
      X1, U1, X1_integration, obj_value = scp_min_xf(robot, robot.X_scpminxf, robot.U_scpminxf, robot.xf, dt, data_neighbors, trust_region=True, trust_x=0.25, trust_u=1, num_iterations=1)
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
      data_neighbors = [(r.cftype, r.X_des) for r in robots[0:idx] + robots[idx+1:]]

      X2, U2, X2_integration, obj_value = scp(robot, robot.X_des, robot.U_des, dt, data_neighbors, trust_region=True, trust_x=0.25, trust_u=1, num_iterations=1)
      robot.X_des = X2[-1]
      robot.U_des = U2[-1]
      robot.obj_values_des.append(obj_value)
      print("... finished with obj value {}".format(obj_value))

  # pickle.dump( robots, open( "robots.p", "wb" ) )

  # robots = pickle.load( open( "robots.p", "rb" ) )

  tracking(robots, dt)
  stats = compute_stats(robots, dt)
  vis_pdf(robots, stats, file_name, use3D=use3D)

  return stats

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--use3D", action='store_true', help="use 3d version")
  args = parser.parse_args()

  sequential_planning(useNN=True, file_name="output.pdf", use3D=args.use3D)