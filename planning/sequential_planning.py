from robots import RobotCrazyFlie2D, RobotCrazyFlie3D
# from sequential_tree_search import tree_search
from sequential_tree_search_ao_rrt import tree_search
# from sequential_tree_search_ao_rrt_scp import tree_search
import torch
import math
from sequential_scp import scp
import numpy as np
import pickle
import subprocess
import random
import argparse
import os
import sys
import logging

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
  fig, ax = plt.subplots(3, 1)
  for k, robot in enumerate(robots):
    X = getattr(robot, 'X_' + field)
    U = getattr(robot, 'U_' + field)
    if use3D:
      line = ax[0].plot(torch.norm(X[:,3:6], dim=1), label="cf{}".format(k))
    else:
      line = ax[0].plot(torch.norm(X[:,2:4], dim=1), label="cf{}".format(k))
    ax[1].plot(torch.norm(U, dim=1), line[0].get_color())
    ax[2].plot(X[:,-1], line[0].get_color())
    colors.append(line[0].get_color())
  ax[0].legend()
  ax[0].set_title('{} - Velocity'.format(name))
  ax[1].legend()
  ax[1].set_title('{} - Acceleration'.format(name))
  ax[2].set_title('{} - Fa [g]'.format(name))
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
  max_z_errors = [0]
  control_efforts = [0]
  stats['tracking_errors_avg'] = 0
  stats['control_efforts_avg'] = 0
  stats['fa_abs_max'] = dict()
  stats['fa_within_bounds'] = True

  for k, robot in enumerate(robots):
    velIdx = robot.X_rollout.shape[1] // 2
    tracking_error = np.sum(np.linalg.norm(robot.X_rollout[:,0:velIdx] - robot.X_des2[:,0:velIdx], axis=1))
    tracking_errors.append(tracking_error)
    tracking_errors[0] += tracking_error
    stats['tracking_errors_avg'] += np.mean(np.linalg.norm(robot.X_rollout[:,0:velIdx] - robot.X_des2[:,0:velIdx], axis=1))

    max_z_error = torch.max(np.abs(robot.X_rollout[:,velIdx-1] - robot.X_des2[:,velIdx-1]))
    max_z_errors.append(max_z_error)
    max_z_errors[0] = max(max_z_errors[0], max_z_error)

    control_effort = np.sum(np.linalg.norm(robot.U_rollout, axis=1)) * dt
    control_efforts.append(control_effort)
    control_efforts[0] += control_effort
    stats['control_efforts_avg'] += np.mean(np.linalg.norm(robot.U_rollout, axis=1))

    if robot.cftype not in stats['fa_abs_max']:
      stats['fa_abs_max'][robot.cftype] = 0

    fa_abs_max = torch.max(torch.abs(robot.Fa_rollout))
    stats['fa_abs_max']["cf{}".format(k)] = fa_abs_max
    stats['fa_abs_max'][robot.cftype] = max(stats['fa_abs_max'][robot.cftype], fa_abs_max)

    epsilon = 0.5 #g
    if torch.max(robot.Fa_rollout) > robot.x_max[-1] + epsilon or \
       torch.min(robot.Fa_rollout) < robot.x_min[-1] - epsilon:
       stats['fa_within_bounds'] = False
       logging.warning("robot {} Fa is out of bounds ({}, {})!".format(k,torch.max(robot.Fa_rollout), torch.min(robot.Fa_rollout)))

  stats['tracking_errors'] = tracking_errors
  stats['max_z_errors'] = max_z_errors
  stats['control_efforts'] = control_efforts
  stats['tracking_errors_avg'] /= len(robots)
  stats['control_efforts_avg'] /= len(robots)

  return stats

def vis_pdf(robots, stats, name='output.pdf', use3D=False, text=None):
  # scp_epoch = len(X)

  pp = PdfPages(name)

  add_result_figs(robots, pp, 'treesearch', 'Tree Search',use3D=use3D)

  if hasattr(robots[0], "X_scpminxf"):
    add_result_figs(robots, pp, 'scpminxf', 'SCP (min xf)',use3D=use3D)

  if hasattr(robots[0], "X_des"):
    add_result_figs(robots, pp, 'des', 'SCP (min u)', use3D=use3D, animate=True)

  if hasattr(robots[0], "X_des2"):

    # roll-out
    if use3D:
      fig = plt.figure()
      ax0 = fig.add_subplot(2,1,1,projection='3d')
      ax1 = fig.add_subplot(2,1,2)
      for k, robot in enumerate(robots):
        X_des2 = np.array(robot.X_des2)
        X_rollout = np.array(robot.X_rollout)
        Fa_rollout = np.array(robot.Fa_rollout)
        line = ax0.plot3D(X_des2[:,0], X_des2[:,1], X_des2[:,2], linestyle='--', label="cf{} des pos".format(k))
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
        X_des2 = np.array(robot.X_des2)
        X_rollout = np.array(robot.X_rollout)
        line = ax.plot3D(X_des2[:,3], X_des2[:,4], X_des2[:,5], linestyle='--', label="cf{} des vel".format(k))
        ax.plot(X_rollout[:,3], X_rollout[:,4], X_rollout[:,5], line[0].get_color(), label="cf{} tracking".format(k))
      set_axes_equal(ax)
      ax.legend()
      ax.set_title('Velocity tracking')
      pp.savefig(fig)
      plt.close(fig)

    else:
      fig, ax = plt.subplots(2, 1)
      for k, robot in enumerate(robots):
        line = ax[0].plot(robot.X_des2[:,0], robot.X_des2[:,1], linestyle='--', label="cf{} des pos".format(k))
        ax[0].plot(robot.X_rollout[:,0], robot.X_rollout[:,1], line[0].get_color(), label="cf{} tracking".format(k))
        ax[1].plot(robot.Fa_rollout, line[0].get_color(), label="cf{} Fa".format(k))
        ax[1].plot(robot.X_des2[:,4], line[0].get_color(), linestyle='--')
      # ax[0].set_aspect('equal')
      ax[0].legend()
      ax[0].set_title('Position tracking')
      ax[1].legend()
      ax[1].set_title('Fa')
      pp.savefig(fig)
      plt.close(fig)

      fig, ax = plt.subplots(2,1)
      for k, robot in enumerate(robots):
        line = ax[0].plot(robot.X_des2[:,2], linestyle='--', label="cf{} des vx".format(k))
        ax[0].plot(robot.X_rollout[:,2], line[0].get_color(), label="cf{} tracking".format(k))
        line = ax[1].plot(robot.X_des2[:,3], linestyle='--', label="cf{} des vz".format(k))
        ax[1].plot(robot.X_rollout[:,3], line[0].get_color(), label="cf{} tracking".format(k))
      ax[0].legend()
      ax[0].set_title('Velocity tracking')
      pp.savefig(fig)
      plt.close(fig)

    fig, ax = plt.subplots(1, 2)
    for k, robot in enumerate(robots):
      if use3D:
        ax[0].plot(robot.U_rollout[:,0], label="cf{} ux".format(k))
        ax[0].plot(robot.U_rollout[:,1], label="cf{} uy".format(k))
        ax[0].plot(robot.U_rollout[:,2], label="cf{} uz".format(k))
      else:
        line = ax[0].plot(robot.U_rollout[:,0], label="cf{} uy".format(k))
        ax[0].plot(robot.U_des2[:,0], line[0].get_color(), linestyle='--')
        line = ax[0].plot(robot.U_rollout[:,1], label="cf{} uz".format(k))
        ax[0].plot(robot.U_des2[:,1], line[0].get_color(), linestyle='--')
      line = ax[1].plot(torch.norm(robot.U_rollout, dim=1), label="cf{} thrust".format(k))
      ax[1].plot(torch.norm(robot.U_des2, dim=1), color=line[0].get_color(), linestyle='--')
      ax[1].axhline(y=robot.g*robot.thrust_to_weight, color=line[0].get_color(), linestyle=':')
    ax[0].legend()  
    ax[1].legend()
    ax[1].set_title('Control and thrust')
    pp.savefig(fig)
    plt.close()

    tracking_errors = stats['tracking_errors']
    control_efforts = stats['control_efforts']
    max_z_errors = stats['max_z_errors']

    fig, ax = plt.subplots(1, 3)
    ax[0].set_title('Tracking errors')
    ax[0].bar(range(len(tracking_errors)), tracking_errors)

    ax[1].set_title('Control efforts')
    ax[1].bar(range(len(control_efforts)), control_efforts)

    ax[2].set_title('Max Z Error')
    ax[2].bar(range(len(max_z_errors)), max_z_errors)

    pp.savefig(fig)
    plt.close()

  if text is not None:
    fig = plt.figure(figsize=(11.69,8.27))
    fig.clf()
    fig.text(0.5,0.5, text, transform=fig.transFigure, size=24, ha="center")
    pp.savefig(fig)
    plt.close()

  pp.close()


def tracking(robots, dt, fixed_T=None):
  # find total time for all robots to reach their goal
  T = 0
  for robot in robots:
    T = max(T, robot.X_des.size(0))

  if fixed_T is not None:
    if T > fixed_T:
      logging.warning("provided fixed time horizon ({}) smaller than planning horizon ({})!".format(fixed_T, T))
    T = fixed_T

  # for each robot allocate the required memory
  for robot in robots:
    robot.X_rollout = torch.zeros((T, robot.stateDim))
    robot.U_rollout = torch.zeros((T-1, robot.ctrlDim))
    robot.Fa_rollout = torch.zeros((T,))
    robot.X_rollout[0] = robot.x0

    robot.X_des2 = torch.zeros((T, robot.stateDim))
    robot.U_des2 = torch.zeros((T-1, robot.ctrlDim))
    robot.X_des2[-1] = robot.xf

    robot.controller_reset()

  # simulate the execution
  with torch.no_grad():
    for t in range(T):
      for k, robot in enumerate(robots):
        # compute neighboring states:
        data_neighbors = []
        for k_other, robot_other in enumerate(robots):
          if k != k_other:
            data_neighbors.append((robot_other.cftype, robot_other.X_rollout[t]))

        # update Fa
        Fa = robot.compute_Fa(robot.X_rollout[t], data_neighbors, useNN_override=True)
        robot.X_rollout[t,4] = Fa
        robot.Fa_rollout[t] = Fa

        # forward propagate
        # Note: We use the "wrong" data_neighbors from the current timestep here
        #       This will be corrected in the next iteration, when Fa is updated
        if t < T-1:
          if t < robot.U_des.size(0):
            x_d = robot.X_des[t]
            u_d = robot.U_des[t]
          else:
            x_d = robot.xf
            u_d = torch.zeros(robot.ctrlDim)
            u_d[-1] = robot.g

          robot.X_des2[t] = x_d
          robot.U_des2[t] = u_d

          robot.U_rollout[t] = robot.controller(x=robot.X_rollout[t], x_d=x_d, v_d_dot=u_d, dt=dt)
          # no need to useNN here, since we'll update Fa anyways
          robot.X_rollout[t+1] = robot.step(robot.X_rollout[t], robot.U_rollout[t], data_neighbors, dt, useNN=False)


def sequential_planning(useNN, robot_info, file_name="output", use3D=False, fixed_T=None, seed=None, log_to_file=True):
  if log_to_file:
    logging.basicConfig(filename='{}.log'.format(file_name),level=logging.INFO)
  else:
    logging.basicConfig(level=logging.INFO)

  if seed is None:
    seed = int.from_bytes(os.urandom(4), sys.byteorder)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  logging.info("seed: {}".format(seed))

  dt = 0.05

  model_folder = '../data/models/val_with23_wdelay/epoch20_lip3_h20_f0d35_B256'
  # model_folder = '../data/models/val_with23_wdelay/epoch20_lip2_h20_f0d35_B256'
  xy_filter = 0.35
  robots = []
  for ri in robot_info:
    if use3D:
      robot = RobotCrazyFlie3D(model_folder, useNN=useNN, cftype=ri['type'], xy_filter=xy_filter)
    else:
      robot = RobotCrazyFlie2D(model_folder, useNN=useNN, cftype=ri['type'], xy_filter=xy_filter)
    robot.x0 = ri['x0']
    robot.xf = ri['xf']
    robots.append(robot)

  logging.info("model: {}".format(model_folder))
  logging.info("robot_info: {}".format(robot_info))

  # Correct initial Fa estimate
  with torch.no_grad():
    for idx, robot in enumerate(robots):
      data_neighbors = [(r.cftype, r.x0) for r in robots[0:idx] + robots[idx+1:]]
      robot.x0[-1] = robot.compute_Fa(robot.x0, data_neighbors)


  # tree search to find initial solution
  while True:

    permutation = list(range(len(robots)))

    if True:

      cost_limits = np.repeat(1e6, len(robots))
      while (cost_limits == 1e6).any():
        cost_limits = np.repeat(1e6, len(robots))
        successive_failures = np.zeros(len(robots))
        for robot in robots:
          if hasattr(robot, "X_treesearch"):
            del robot.X_treesearch
          if hasattr(robot, "U_treesearch"):
            del robot.U_treesearch

        # for trial in range(100):
        trial = 0
        while (successive_failures < 2).any():
        # while (cost_limits == 1e6).any():
          trial += 1
          logging.info("Tree search trial {}; successive_failures: {}; cost_limit: {}, s: {}".format(trial, successive_failures, cost_limits, seed))
          random.shuffle(permutation)
          for idx in permutation:
            logging.info("Tree search for robot {}".format(idx))
            robot = robots[idx]
            data_neighbors = [(r.cftype, r.X_treesearch) for r in robots[0:idx] + robots[idx+1:] if hasattr(r, 'X_treesearch')]

            # run tree search to find initial solution
            with torch.no_grad():
              if use3D:
                x, u, best_cost = tree_search(robot, robot.x0, robot.xf, dt, data_neighbors, prop_iter=1, iters=500000, top_k=10, num_branching=2, trials=1, cost_limit=cost_limits[idx])
              else:
                x, u, best_cost = tree_search(robot, robot.x0, robot.xf, dt, data_neighbors, prop_iter=1, iters=100000, top_k=10, num_branching=2, trials=1, cost_limit=cost_limits[idx])
              
              if x is None:
                successive_failures[idx] += 1
              else:
                # found a new solution
                cost_limits[idx] = 0.9 * best_cost
                successive_failures[idx] = 0
                robot.X_treesearch = x
                robot.U_treesearch = u

                # # this new solution can affect Fa of neighbors and therefore the dynamics
                # # of the neighbors.
                # # Do a sanity check of the neighboring trajectories and invalidate them if needed
                # for idx_other, robot_other in enumerate(robots):
                #   robot_other = robots[idx_other]
                #   if idx != idx_other and hasattr(robot_other, 'X_treesearch'):
                #     print("Checking if robot {} is still valid.".format(idx_other))
                #     T_other = robot_other.X_treesearch.size(0)
                #     for t in range(0,T_other-1):
                #       # compute neighboring states:
                #       data_neighbors_next = []
                #       for r in robots[0:idx_other] + robots[idx_other+1:]:
                #         if hasattr(r, 'X_treesearch'):
                #           if t+1 < r.X_treesearch.size(0):
                #             data_neighbors_next.append((r.cftype, r.X_treesearch[t+1]))
                #           else:
                #             data_neighbors_next.append((r.cftype, r.X_treesearch[-1]))

                #       # propagate dynamics and check
                #       X_next = robot_other.step(robot_other.X_treesearch[t], robot_other.U_treesearch[t], data_neighbors_next, dt)

                #       # if not torch.isclose(robot_other.X_treesearch[t+1], X_next).all():
                #       if not abs(X_next[4] - robot_other.X_treesearch[t+1,4]) < 5:
                #         print("WARNING: trajectory of robot {} at t={} now invalid.".format(idx_other, t))
                #         successive_failures[idx_other] = 0
                #         cost_limits[idx_other] = 1e6
                #         del robot_other.X_treesearch
                #         del robot_other.U_treesearch
                #         break

      pickle.dump( robots, open( "treesearch.p", "wb" ) )
    else:
      robots = pickle.load( open( "treesearch.p", "rb" ) )
      # robots = pickle.load( open( "/home/whoenig/projects/caltech/neural-swarm/data/planning/case_SL_iter_4_NN.pdf.p", "rb" ) )

    # vis_pdf(robots, "output.pdf", use3D=use3D)
    # exit()

    # # resize all X/U_scpminxf in order to avoid corner cases where one robot stays at its goal and is subject
    # # to high Fa
    # T = 0
    # for robot in robots:
    #   T = max(T, robot.X_treesearch.size(0))

    # # virtually track the trajectory to make sure the dynamics
    # # consider the inter-robot interaction correctly

    # # for each robot allocate the required memory
    # for robot in robots:
    #   robot.X_treesearch_rollout = torch.zeros((T, robot.stateDim))
    #   robot.X_treesearch_rollout[0] = robot.x0
    #   robot.U_treesearch_rollout = torch.zeros((T-1, robot.ctrlDim))

    #   # extend/shorten X_des and U_des
    #   robot.X_treesearch_des = torch.zeros((T, robot.stateDim))
    #   robot.U_treesearch_des = torch.zeros((T-1, robot.ctrlDim))
    #   max_idx = min(robot.X_treesearch.size(0), T)
    #   robot.X_treesearch_des[0:max_idx] = robot.X_treesearch[0:max_idx]
    #   robot.X_treesearch_des[max_idx:] = robot.X_treesearch[-1]
    #   max_idx = min(robot.U_treesearch.size(0), T-1)
    #   robot.U_treesearch_des[0:max_idx] = robot.U_treesearch[0:max_idx]
    #   robot.U_treesearch_des[max_idx:] = torch.zeros(robot.ctrlDim)
    #   robot.U_treesearch_des[max_idx:,-1] = robot.g

    # # simulate the execution
    # with torch.no_grad():
    #   for t in range(T-1):
    #     for k, robot in enumerate(robots):
    #       # compute neighboring states:
    #       data_neighbors = []
    #       for k_other, robot_other in enumerate(robots):
    #         if k != k_other:
    #           data_neighbors.append((robot_other.cftype, robot_other.X_treesearch_rollout[t]))

    #       Fa = robot.compute_Fa(robot.X_treesearch_rollout[t], data_neighbors, True)

    #       # control + forward propagation

    #       # if the planner did not account for this part and Fa prediction is enabled
    #       # correct the feedforward term
    #       max_idx = min(robot.U_treesearch.size(0), T-1)
    #       if t >= max_idx and robot.useNN:
    #         robot.U_treesearch_des[t,-1] -= robot.g*Fa/robot.mass

    #       v_d_dot = robot.U_treesearch_des[t]

    #       x_d = robot.X_treesearch_des[t]

    #       u = robot.controller(x=robot.X_treesearch_rollout[t], x_d=x_d, v_d_dot=v_d_dot, dt=dt)
    #       robot.U_treesearch_rollout[t] = u
    #       dx = robot.f(robot.X_treesearch_rollout[t], robot.U_treesearch_rollout[t], data_neighbors, useNN=True)
    #       robot.X_treesearch_rollout[t+1] = robot.X_treesearch_rollout[t] + dt*dx

    # # Initialize X_scpminxf to SCP
    # for k, robot in enumerate(robots):
    #   robot.X_scpminxf = robot.X_treesearch_rollout.clone()
    #   robot.U_scpminxf = robot.U_treesearch_rollout.clone()
    #   robot.obj_values_scpminxf = []

    # # Initialize X_scpminxf to SCP
    # for k, robot in enumerate(robots):
    #   robot.X_scpminxf = robot.X_treesearch.clone()
    #   robot.U_scpminxf = robot.U_treesearch.clone()
    #   robot.obj_values_scpminxf = []

    # vis_pdf(robots, "output.pdf", use3D=use3D)
    # exit()

    # # run scp (min xf) to exactly go to the goal
    trust_x = [0.05,0.05,0.05,0.05,0]
    trust_u = 1
    # for iteration in range(100):
    #   random.shuffle(permutation)
    #   all_robots_done = True
    #   for idx in permutation:
    #     print("SCP (min xf) for robot {}".format(idx))
    #     robot = robots[idx]
    #     data_neighbors = [(r.cftype, r.X_scpminxf) for r in robots[0:idx] + robots[idx+1:]]
    #     X1, U1, X1_integration, obj_value = scp_min_xf(robot, robot.X_scpminxf, robot.U_scpminxf, robot.xf, dt, data_neighbors, trust_region=True, trust_x=trust_x, trust_u=trust_u, num_iterations=1)
    #     robot.X_scpminxf = X1[-1]
    #     robot.U_scpminxf = U1[-1]
    #     robot.obj_values_scpminxf.append(obj_value)
    #     print("... finished with obj value {}".format(obj_value))
    #     # setup for next stage
    #     robot.X_des = robot.X_scpminxf
    #     robot.U_des = robot.U_scpminxf
    #     robot.obj_values_des = []
    #     if obj_value > 1e-8:
    #       all_robots_done = False
    #   if all_robots_done:
    #     break

    # vis_pdf(robots, "output.pdf", use3D=use3D)
    # exit()

    postprocessing = 'rollout'

    if postprocessing == 'none':
      # Initialize X_scpminxf to SCP
      for k, robot in enumerate(robots):
        robot.X_des = robot.X_treesearch.clone()
        robot.U_des = robot.U_treesearch.clone()
        robot.obj_values_des = []

    elif postprocessing == 'resizeT':
      # resize all X/U_scpminxf in order to avoid corner cases where one robot stays at its goal and is subject
      # to high Fa
      T = 0
      for robot in robots:
        T = max(T, robot.X_treesearch.size(0))

      for robot in robots:
        robot.X_des = torch.zeros((T, robot.stateDim))
        robot.U_des = torch.zeros((T-1, robot.ctrlDim))
        robot.obj_values_des = []

        # extend/shorten X_des and U_des
        max_idx = min(robot.X_treesearch.size(0), T)
        robot.X_des[0:max_idx] = robot.X_treesearch[0:max_idx]
        robot.X_des[max_idx:] = robot.X_treesearch[-1]
        max_idx = min(robot.U_treesearch.size(0), T-1)
        robot.U_des[0:max_idx] = robot.U_treesearch[0:max_idx]
        robot.U_des[max_idx:] = torch.zeros(robot.ctrlDim)
        robot.U_des[max_idx:,-1] = robot.g

    elif postprocessing == 'rollout':

      # resize all X/U_scpminxf in order to avoid corner cases where one robot stays at its goal and is subject
      # to high Fa
      T = 0
      for robot in robots:
        T = max(T, robot.X_treesearch.size(0))

      old_dt = dt
      new_dt = 0.05

      T = T * old_dt
      steps = int(T/new_dt)

      for robot in robots:
        robot.X_des = torch.zeros((steps, robot.stateDim))
        robot.X_des[0] = robot.X_treesearch[0]
        robot.U_des = torch.zeros((steps-1, robot.ctrlDim))
        robot.obj_values_des = []

        robot.controller_reset()

        # # extend/shorten X_des and U_des
        # max_idx = min(robot.X_treesearch.size(0), T)
        # robot.X_des[0:max_idx] = robot.X_treesearch[0:max_idx]
        # robot.X_des[max_idx:] = robot.X_treesearch[-1]
        # max_idx = min(robot.U_treesearch.size(0), T-1)
        # robot.U_des[0:max_idx] = robot.U_treesearch[0:max_idx]
        # robot.U_des[max_idx:] = torch.zeros(robot.ctrlDim)
        # robot.U_des[max_idx:,-1] = robot.g

      # simulate the execution
      with torch.no_grad():
        for t in range(steps):
          for k, robot in enumerate(robots):
            # compute neighboring states:
            data_neighbors = []
            for k_other, robot_other in enumerate(robots):
              if k != k_other:
                data_neighbors.append((robot_other.cftype, robot_other.X_des[t]))

            # update Fa
            Fa = robot.compute_Fa(robot.X_des[t], data_neighbors)
            robot.X_des[t,4] = Fa

            # forward propagate
            # Note: We use the "wrong" data_neighbors from the current timestep here
            #       This will be corrected in the next iteration, when Fa is updated
            idx_old = math.floor(t*new_dt/old_dt)
            if t < steps-1:
              if idx_old < robot.X_treesearch.size(0) - 2:
                weight = t*new_dt/old_dt - idx_old
                x_d = torch.lerp(robot.X_treesearch[idx_old], robot.X_treesearch[idx_old+1], weight)
                u_d = torch.lerp(robot.U_treesearch[idx_old], robot.U_treesearch[idx_old+1], weight)
              else:
                x_d = robot.xf
                u_d = torch.zeros(robot.ctrlDim)
                u_d[-1] = robot.g

              robot.U_des[t] = robot.controller(x=robot.X_des[t], x_d=x_d, v_d_dot=u_d, dt=new_dt)
              robot.X_des[t+1] = robot.step(robot.X_des[t], robot.U_des[t], data_neighbors, new_dt)

      # update dt
      dt = new_dt

    # # Half timesteps for optimization
    # with torch.no_grad():
    #   # for each robot allocate the required memory
    #   for idx, robot in enumerate(robots):
    #     robot.X_des = torch.zeros((robot.X_treesearch.size(0)*2-1, robot.stateDim))
    #     robot.U_des = torch.zeros((robot.X_treesearch.size(0)*2-2, robot.ctrlDim))
    #     T = robot.X_treesearch.size(0)
    #     for t in range(T-1):
    #       # compute neighboring states:
    #       data_neighbors = []
    #       for r in robots[0:idx] + robots[idx+1:]:
    #         if t < r.X_treesearch.size(0):
    #           data_neighbors.append((r.cftype, r.X_treesearch[t]))
    #         else:
    #           data_neighbors.append((r.cftype, r.X_treesearch[-1]))

    #       # propagate dynamics
    #       X_next = robot.step(robot.X_treesearch[t], robot.U_treesearch[t], data_neighbors, dt/2)

    #       robot.X_des[2*t] = robot.X_treesearch[t]
    #       robot.X_des[2*t+1] = X_next

    #       robot.U_des[2*t] = robot.U_treesearch[t]
    #       robot.U_des[2*t+1] = robot.U_treesearch[t]
    #     robot.X_des[-1] = robot.X_treesearch[-1]

    # dt = dt/2

    # vis_pdf(robots, "output.pdf", use3D=use3D)
    # exit()

    # run scp to minimize U
    for iteration in range(30):
      random.shuffle(permutation)
      failure = False
      for idx in permutation:
        logging.info("SCP (min u) for robot {}".format(idx))
        robot = robots[idx]
        data_neighbors = [(r.cftype, r.X_des) for r in robots[0:idx] + robots[idx+1:]]

        # update trust region for Fa
        trust_x[-1] = robot.trust_Fa(robot.cftype)
        # run optimization
        X2, U2, X2_integration, obj_value = scp(robot, robot.X_des, robot.U_des, robot.xf, dt, data_neighbors, trust_region=True, trust_x=trust_x, trust_u=trust_u, num_iterations=1)
        robot.obj_values_des.append(obj_value)
        logging.info("... finished with obj value {}".format(obj_value))
        if obj_value != np.inf:
          robot.X_des = X2[-1]
          robot.U_des = U2[-1]
        else:
          failure = True
        # else:
        #   print("Re-run min xf for robot {}".format(idx))
        #   while True:
        #     X2, U2, X2_integration, obj_value = scp_min_xf(robot, robot.X_des, robot.U_des, robot.xf, dt, data_neighbors, trust_region=True, trust_x=[0.02,0.02,0.1,0.1,1.2], trust_u=1, num_iterations=1)
        #     print("... finished with obj value {}".format(obj_value))
        #     if obj_value != np.inf:
        #       robot.X_des = X2[-1]
        #       robot.U_des = U2[-1]
        #     if obj_value < 1e-8:
        #       break

    # try again if optimization failed
    failure = False
    for robot in robots:
      if np.isinf(robot.obj_values_des).all():
        failure = True
        break
    if failure:
      continue
    else:
      pickle.dump( robots, open( "{}.p".format(file_name), "wb" ) )
      break

  tracking(robots, dt, fixed_T=fixed_T)
  stats = compute_stats(robots, dt)
  vis_pdf(robots, stats, "{}.pdf".format(file_name), use3D=use3D)

  return stats

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--use3D", action='store_true', help="use 3d version")
  parser.add_argument("--vis", help="visualize a *.p file")
  parser.add_argument("--seed", type=int, default=None, help="random number seed")
  args = parser.parse_args()

  if args.vis:
    dt = 0.05
    robots = pickle.load(open(args.vis, "rb"))
    tracking(robots, dt)
    stats = compute_stats(robots, dt)
    vis_pdf(robots, stats, "output.pdf", use3D=args.use3D)
  else:

    if args.use3D:
      robot_info = [
          {
            'type': 'small',
            'x0': torch.tensor([0,-0.5,0,0,0,0], dtype=torch.float32),
            'xf': torch.tensor([0,0.5,0,0,0,0], dtype=torch.float32),
          },
          {
            'type': 'large',
            'x0': torch.tensor([0,0.5,0,0,0,0], dtype=torch.float32),
            'xf': torch.tensor([0,-0.5,0,0,0,0], dtype=torch.float32),
          },
        ]
    else:
      robot_info = [
          {
            'type': 'small',
            'x0': torch.tensor([-0.3,1,0,0,0], dtype=torch.float32),
            'xf': torch.tensor([0.3,1,0,0,0], dtype=torch.float32),
          },
          {
            'type': 'large',
            # 'type': 'small',
            'x0': torch.tensor([0.3,1,0,0,0], dtype=torch.float32),
            'xf': torch.tensor([-0.3,1,0,0,0], dtype=torch.float32),
          },
          # {
          #   'type': 'small',
          #   'x0': torch.tensor([0.0,1,0,0,0], dtype=torch.float32),
          #   'xf': torch.tensor([0.6,1,0,0,0], dtype=torch.float32),
          # },
          # {
          #   'type': 'small',
          #   'x0': torch.tensor([0.6,1,0,0,0], dtype=torch.float32),
          #   'xf': torch.tensor([0.0,1,0,0,0], dtype=torch.float32),
          # },
        ]

    sequential_planning(True, robot_info, file_name="output", use3D=args.use3D, seed=args.seed, log_to_file=False)

  subprocess.call(["xdg-open", "output.pdf"])