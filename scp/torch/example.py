from robots import RobotDubinsCar, RobotDoubleIntegrator, RobotAirplane, RobotTwoCrazyFlies2D
import torch
import math
from scp import scp
import numpy as np
import subprocess

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def linspace(start, stop, steps):
  return torch.stack([torch.linspace(start[i], stop[i], steps) for i in range(start.size(0))],dim=1)

def distance(x):
  dist = torch.zeros(x.size(0))
  for i in range(x.size(0)):
    dist[i] = torch.norm(torch.tensor([x[i,0]-x[i,4], x[i,1]-x[i,5]]))
  return dist

def thrust(u):
  return torch.stack([torch.norm(u[i,:2]) for i in range(u.size(0))]), torch.stack([torch.norm(u[i,2:]) for i in range(u.size(0))]) 

def vis(robot, initial_x, initial_u, X, U, X_integration, plot_integration=False):
  scp_epoch = len(X)

  # y-z trajectory and distance
  fig, ax = plt.subplots(2, scp_epoch+1)
  ax[0,0].plot(initial_x[:,0], initial_x[:,1], label="cf1 initial")
  ax[0,0].plot(initial_x[:,4], initial_x[:,5], label="cf2 initial")
  ax[0,0].legend()
  for i in range(scp_epoch):
    ax[0,i+1].plot(X[i][:,0], X[i][:,1], label="cf1 e" + str(i+1))
    ax[0,i+1].plot(X[i][:,4], X[i][:,5], label="cf2 e" + str(i+1))
    if plot_integration:
      ax[0,i+1].plot(X_integration[i][:,0], X_integration[i][:,1], label="int cf1 e" + str(i+1))
      ax[0,i+1].plot(X_integration[i][:,4], X_integration[i][:,5], label="int cf2 e" + str(i+1))
    ax[0,i+1].legend()
  plt.title('y-z trajectory')
  ax[1,0].plot(distance(initial_x), label="dist initial")
  ax[1,0].axhline(y=2*robot.radius, linestyle='--', label="limit")
  ax[1,0].legend()
  for i in range(scp_epoch):
    ax[1,i+1].plot(distance(X[i]), label="dist e" + str(i+1))
    ax[1,i+1].axhline(y=2*robot.radius, linestyle='--', label="limit")
    ax[1,i+1].legend()
  plt.title('distance')
  # plt.savefig('fig-1.png')
  # plt.show()

  # control and thrust magnitude
  fig, ax = plt.subplots(2, scp_epoch+1)
  ax[0,0].plot(initial_u[:,0], label="cf1 uy init")
  ax[0,0].plot(initial_u[:,1], label="cf1 uz init")
  ax[0,0].plot(initial_u[:,2], label="cf2 uy init")
  ax[0,0].plot(initial_u[:,3], label="cf2 uz init")
  ax[0,0].legend()
  for i in range(scp_epoch):
    ax[0,i+1].plot(U[i][:,0], label="cf1 uy e" + str(i+1))
    ax[0,i+1].plot(U[i][:,1], label="cf1 uz e" + str(i+1))
    ax[0,i+1].plot(U[i][:,2], label="cf2 uy e" + str(i+1))
    ax[0,i+1].plot(U[i][:,3], label="cf2 uz e" + str(i+1))
    ax[0,i+1].legend()
  plt.title('control')
  ax[1,0].plot(thrust(initial_u)[0], label="cf1 thrust init")
  ax[1,0].plot(thrust(initial_u)[1], label="cf2 thrust init")
  ax[1,0].axhline(y=robot.g*robot.thrust_to_weight, linestyle='--', label="limit")
  ax[1,0].legend()
  for i in range(scp_epoch):
    ax[1,i+1].plot(thrust(U[i])[0], label="cf1 thrust e" + str(i+1))
    ax[1,i+1].plot(thrust(U[i])[1], label="cf2 thrust e" + str(i+1))
    ax[1,i+1].axhline(y=robot.g*robot.thrust_to_weight, linestyle='--', label="limit")
    ax[1,i+1].legend()
  plt.title('thrust magnitude')
  # plt.savefig('fig-2.png')
  # plt.show()

  # velocity
  fig, ax = plt.subplots(1, scp_epoch+1)
  ax[0].plot(initial_x[:,2], label="cf1 vy init")
  ax[0].plot(initial_x[:,3], label="cf1 vz init")
  ax[0].plot(initial_x[:,6], label="cf2 vy init")
  ax[0].plot(initial_x[:,7], label="cf2 vz init")
  ax[0].legend()
  for i in range(scp_epoch):
    ax[i+1].plot(X[i][:,2], label="cf1 vy e" + str(i+1))
    ax[i+1].plot(X[i][:,3], label="cf1 vz e" + str(i+1))
    ax[i+1].plot(X[i][:,6], label="cf2 vy e" + str(i+1))
    ax[i+1].plot(X[i][:,7], label="cf2 vz e" + str(i+1))
    ax[i+1].legend()
  plt.title('velocity')
  # plt.savefig('fig-3.png')
  plt.show()

def vis_pdf(robot, initial_x, initial_u, X, U, X_integration, plot_integration=False):
  scp_epoch = len(X)

  pp = PdfPages("output.pdf")

  # initial y-z trajectory
  fig, ax = plt.subplots()
  ax.axis('equal')
  ax.plot(initial_x[:,0], initial_x[:,1], label="cf1")
  ax.plot(initial_x[:,4], initial_x[:,5], label="cf2")
  ax.legend()
  ax.set_title('y-z trajectory (initial)')
  pp.savefig(fig)
  plt.close(fig)

  # initial distance
  fig, ax = plt.subplots()
  ax.plot(distance(initial_x), label="distance")
  ax.axhline(y=2*robot.radius, linestyle='--', label="limit")
  ax.legend()
  ax.set_title('distance (initial)')
  pp.savefig(fig)
  plt.close(fig)

  # control
  fig, ax = plt.subplots()
  ax.plot(initial_u[:,0], label="cf1 uy")
  ax.plot(initial_u[:,1], label="cf1 uz")
  ax.plot(initial_u[:,2], label="cf2 uy")
  ax.plot(initial_u[:,3], label="cf2 uz")
  ax.legend()
  ax.set_title('control (initial)')
  pp.savefig(fig)
  plt.close(fig)

  # thrust magnitude
  fig, ax = plt.subplots()
  ax.plot(thrust(initial_u)[0], label="cf1")
  ax.plot(thrust(initial_u)[1], label="cf2")
  ax.axhline(y=robot.g*robot.thrust_to_weight, linestyle='--', label="limit")
  ax.legend()
  ax.set_title('thrust magnitude (initial)')
  pp.savefig(fig)
  plt.close(fig)

  # velocity
  fig, ax = plt.subplots()
  ax.plot(initial_x[:,2], label="cf1 vy")
  ax.plot(initial_x[:,3], label="cf1 vz")
  ax.plot(initial_x[:,6], label="cf2 vy")
  ax.plot(initial_x[:,7], label="cf2 vz")
  ax.legend()
  ax.set_title('velocity (initial)')
  pp.savefig(fig)
  plt.close(fig)

  for i in range(scp_epoch):
    # y-z trajectory and distance
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.plot(X[i][:,0], X[i][:,1], label="cf1")
    ax.plot(X[i][:,4], X[i][:,5], label="cf2")
    if plot_integration:
      ax.plot(X_integration[i][:,0], X_integration[i][:,1], label="cf1 integration")
      ax.plot(X_integration[i][:,4], X_integration[i][:,5], label="cf2 integration")
    ax.legend()
    ax.set_title('y-z trajectory (iter {})'.format(i))
    pp.savefig(fig)
    plt.close(fig)

    # distance
    fig, ax = plt.subplots()
    ax.plot(distance(X[i]), label="dist initial")
    ax.axhline(y=2*robot.radius, linestyle='--', label="limit")
    ax.legend()
    ax.set_title('distance (iter {})'.format(i))
    pp.savefig(fig)
    plt.close(fig)

    # control
    fig, ax = plt.subplots()
    ax.plot(U[i][:,0], label="cf1 uy")
    ax.plot(U[i][:,1], label="cf1 uz")
    ax.plot(U[i][:,2], label="cf2 uy")
    ax.plot(U[i][:,3], label="cf2 uz")
    ax.legend()
    ax.set_title('control (iter {})'.format(i))
    pp.savefig(fig)
    plt.close(fig)

    # thrust magnitude
    fig, ax = plt.subplots()
    ax.plot(thrust(U[i])[0], label="cf1")
    ax.plot(thrust(U[i])[1], label="cf2")
    ax.axhline(y=robot.g*robot.thrust_to_weight, linestyle='--', label="limit")
    ax.legend()
    ax.set_title('thrust magnitude (iter {})'.format(i))
    pp.savefig(fig)
    plt.close(fig)

    # velocity
    fig, ax = plt.subplots()
    ax.plot(X[i][:,2], label="cf1 vy")
    ax.plot(X[i][:,3], label="cf1 vz")
    ax.plot(X[i][:,6], label="cf2 vy")
    ax.plot(X[i][:,7], label="cf2 vz")
    ax.legend()
    ax.set_title('velocity (iter {})'.format(i))
    pp.savefig(fig)
    plt.close(fig)

  pp.close()
  subprocess.call(["xdg-open", "output.pdf"])


if __name__ == '__main__':
  # # robot = RobotDubinsCar(1, 1)

  # # x = torch.tensor([1,2,3], dtype=torch.float32)
  # # u = torch.tensor([-0.5, 0.7], dtype=torch.float32)

  # # print(robot.f(x, u))

  # robot = RobotAirplane()

  # x = torch.tensor(robot.x_min, dtype=torch.float32)
  # u = torch.tensor(robot.u_min, dtype=torch.float32)

  # print(robot.f(x, u))

  # A, B = jacobian(robot, x, u)
  # print('A', A)
  # print('B', B)

  # robot = RobotDubinsCar(1, 1)

  # x0 = torch.tensor([30,30,0], dtype=torch.float32)
  # xf = torch.tensor([35,30,0], dtype=torch.float32)
  # dt = 0.1
  # T = 10
  # num_steps = int(T / dt)

  # robot = RobotDoubleIntegrator()

  # x0 = torch.tensor([0,0,0,0], dtype=torch.float32)
  # xf = torch.tensor([1,2,0,0], dtype=torch.float32)
  # dt = 0.1
  # T = 10
  # num_steps = int(T / dt)

  # initial_x = linspace(x0, xf, num_steps)
  # initial_u = torch.zeros((num_steps, robot.ctrlDim))

  # x, u = scp(robot, initial_x, initial_u, dt, num_iterations=2)

  robot = RobotTwoCrazyFlies2D(useNN=False)
  robot_NN = RobotTwoCrazyFlies2D(useNN=True)
  # w/o NN: T=2,3, dt=0.1, trust_x=2, trust_u=2,3 will work!
  # w/ NN: T=2,3, dt=0.1, trust_x=2, trust_u=2 will work!

  x0 = torch.tensor([-0.5,0,0,0,0.5,0,0,0], dtype=torch.float32)
  xf = torch.tensor([0.5,0,0,0,-0.5,0,0,0], dtype=torch.float32)
  dt = 0.1
  T = 2
  num_steps = int(T / dt)

  initial_x = linspace(x0, xf, num_steps)
  # better initial_x
  theta = np.linspace(np.pi, 2*np.pi, num_steps)
  for i in range(num_steps):
    initial_x[i, 0] = 0.5*np.cos(theta[i])
    initial_x[i, 1] = -0.5*np.sin(theta[i])
    initial_x[i, 4] = -0.5*np.cos(theta[i])
    initial_x[i, 5] = 0.5*np.sin(theta[i])    

  initial_u = torch.zeros((num_steps, robot.ctrlDim))
  initial_u[:, [1,3]] = robot.g

  scp_epoch = 10
  # X, U, X_integration = scp(robot, initial_x, initial_u, dt, trust_region=True, trust_x=2, trust_u=2, num_iterations=scp_epoch)
  X_NN, U_NN, X_integration_NN = scp(robot_NN, initial_x, initial_u, dt, trust_region=True, trust_x=2, trust_u=3, num_iterations=scp_epoch)

  # vis(robot, initial_x, initial_u, X, U, X_integration, plot_integration=False)
  # vis(robot_NN, initial_x, initial_u, X_NN, U_NN, X_integration_NN, plot_integration=False)
  vis_pdf(robot_NN, initial_x, initial_u, X_NN, U_NN, X_integration_NN, plot_integration=False)
