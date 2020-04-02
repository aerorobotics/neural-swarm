from robots import RobotDubinsCar, RobotDoubleIntegrator, RobotAirplane, RobotTwoCrazyFlies2D
import torch
import math
from scp import scp, scp_sequential, scp_sequential_2
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

def vis(robot, initial_x, initial_u, X, U, X_integration, x_d, x_rollout, u_rollout, plot_integration=False):
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

  # roll-out
  fig, ax = plt.subplots(1, 4)
  ax[0].plot(x_d[:,0], x_d[:,1], 'g', linestyle='--', label="cf1 des pos")
  ax[0].plot(x_rollout[:,0], x_rollout[:,1], 'g', label="cf1 tracking")
  ax[0].plot(x_d[:,4], x_d[:,5], 'b', linestyle='--', label="cf2 des pos")
  ax[0].plot(x_rollout[:,4], x_rollout[:,5], 'b', label="cf2 tracking")
  ax[0].legend()
  ax[1].plot(x_d[:,2], x_d[:,3], 'g', linestyle='--', label="cf1 des vel")
  ax[1].plot(x_rollout[:,2], x_rollout[:,3], 'g', label="cf1 tracking")
  ax[1].plot(x_d[:,6], x_d[:,7], 'b', linestyle='--', label="cf2 des vel")
  ax[1].plot(x_rollout[:,6], x_rollout[:,7], 'b', label="cf2 tracking")
  ax[1].legend()
  ax[2].plot(u_rollout[:,0], label="cf1 uy")
  ax[2].plot(u_rollout[:,1], label="cf1 uz")
  ax[2].plot(u_rollout[:,2], label="cf2 uy")
  ax[2].plot(u_rollout[:,3], label="cf2 uz")
  ax[2].legend()  
  ax[3].plot(thrust(u_rollout)[0], label="cf1 thrust")
  ax[3].plot(thrust(u_rollout)[1], label="cf2 thrust")
  ax[3].axhline(y=robot.g*robot.thrust_to_weight, linestyle='--', label="limit")
  ax[3].legend()
  plt.title('roll-out')
  plt.show()

def vis_pdf(robot, initial_x, initial_u, X, U, X_integration, x_d, x_rollout, u_rollout, f_a, plot_integration=False, name='outpuf.pdf'):
  scp_epoch = len(X)

  pp = PdfPages(name)

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

  # roll-out
  fig, ax = plt.subplots(2, 1)
  ax[0].plot(x_d[:,0], x_d[:,1], 'g', linestyle='--', label="cf1 des pos")
  ax[0].plot(x_rollout[:,0], x_rollout[:,1], 'g', label="cf1 tracking")
  ax[0].plot(x_d[:,4], x_d[:,5], 'b', linestyle='--', label="cf2 des pos")
  ax[0].plot(x_rollout[:,4], x_rollout[:,5], 'b', label="cf2 tracking")
  ax[0].legend()
  ax[0].set_title('Position tracking')
  ax[1].plot(f_a[:,0], 'g', label="cf1 Fa")
  ax[1].plot(f_a[:,1], 'b', label="cf2 Fa")
  ax[1].legend()
  ax[1].set_title('Fa') 
  pp.savefig(fig)
  plt.close(fig)

  fig, ax = plt.subplots()
  ax.plot(x_d[:,2], x_d[:,3], 'g', linestyle='--', label="cf1 des vel")
  ax.plot(x_rollout[:,2], x_rollout[:,3], 'g', label="cf1 tracking")
  ax.plot(x_d[:,6], x_d[:,7], 'b', linestyle='--', label="cf2 des vel")
  ax.plot(x_rollout[:,6], x_rollout[:,7], 'b', label="cf2 tracking")
  ax.legend()
  ax.set_title('Velocity tracking')
  pp.savefig(fig)
  plt.close(fig)

  fig, ax = plt.subplots(1, 2)
  ax[0].plot(u_rollout[:,0], label="cf1 uy")
  ax[0].plot(u_rollout[:,1], label="cf1 uz")
  ax[0].plot(u_rollout[:,2], label="cf2 uy")
  ax[0].plot(u_rollout[:,3], label="cf2 uz")
  ax[0].legend()  
  ax[1].plot(thrust(u_rollout)[0], label="cf1 thrust")
  ax[1].plot(thrust(u_rollout)[1], label="cf2 thrust")
  ax[1].axhline(y=robot.g*robot.thrust_to_weight, linestyle='--', label="limit")
  ax[1].legend()
  ax[1].set_title('Control and thrust')
  pp.savefig(fig)
  plt.close()

  pp.close()
  subprocess.call(["xdg-open", name])

def tracking(robot, dt, x0, X_d, feedforward=True, ctrl_useNN=False):
  X = torch.zeros((X_d.size(0), robot.stateDim))
  U = torch.zeros((X_d.size(0)-1, robot.ctrlDim))
  Fa = torch.zeros((X_d.size(0)-1, 2))
  X[0] = x0
  for i in range(X_d.size(0)-1):
    if feedforward:
      v_d_dot = torch.stack([
        (X_d[i+1, 2]-X_d[i, 2]) / dt,
        (X_d[i+1, 3]-X_d[i, 3]) / dt,
        (X_d[i+1, 6]-X_d[i, 6]) / dt,
        (X_d[i+1, 7]-X_d[i, 7]) / dt])
    else:
      v_d_dot = torch.zeros(robot.ctrlDim)
    u = robot.controller(x=X[i], x_d=X_d[i], v_d_dot=v_d_dot, ctrl_useNN=ctrl_useNN)
    U[i] = u
    dx, fa = robot.f(X[i], U[i], eva_Fa=True)
    X[i+1] = X[i] + dt*dx
    Fa[i] = fa
  return X.detach(), U.detach(), Fa.detach()

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

  '''
  scp_epoch = 10
  X, U, X_integration = scp(robot, initial_x, initial_u, dt, trust_region=True, trust_x=2, trust_u=3, num_iterations=scp_epoch)
  X_NN, U_NN, X_integration_NN = scp(robot_NN, initial_x, initial_u, dt, trust_region=True, trust_x=2, trust_u=3, num_iterations=scp_epoch)

  # in roll-out, we always use robot_NN!
  x_rollout, u_rollout, fa = tracking(robot_NN, dt, x0, X_d=X[-1], feedforward=True, ctrl_useNN=False)
  x_rollout_NN, u_rollout_NN, fa_NN = tracking(robot_NN, dt, x0, X_d=X_NN[-1], feedforward=True, ctrl_useNN=False)

  # vis(robot, initial_x, initial_u, X, U, X_integration, x_d=X[-1], x_rollout=x_rollout, u_rollout=u_rollout, plot_integration=False)
  # vis(robot_NN, initial_x, initial_u, X_NN, U_NN, X_integration_NN, x_d=X_NN[-1], x_rollout=x_rollout_NN, u_rollout=u_rollout_NN, plot_integration=False)
  vis_pdf(robot, initial_x, initial_u, X, U, X_integration, X[-1], x_rollout, u_rollout, fa, plot_integration=False, name='PlanwoNN.pdf')
  vis_pdf(robot_NN, initial_x, initial_u, X_NN, U_NN, X_integration_NN, X_NN[-1], x_rollout_NN, u_rollout_NN, fa_NN, plot_integration=False, name='PlanwithNN.pdf')
  '''

  ################# Try sequential SCP #################
  # warm-up
  scp_epoch = 2
  X_warm, U_warm, X_integration_warm = scp(robot, initial_x, initial_u, dt, trust_region=True, trust_x=2, trust_u=3, num_iterations=scp_epoch)

  scp_epoch = 4
  # Note: both scp_sequential and scp_sequential_2 will converge to some "bad" solutions...
  # X, U, X_integration = scp_sequential(robot, X_warm[-1], U_warm[-1], dt, trust_region=True, trust_x=2, trust_u=3, num_iterations=scp_epoch)
  X, U, X_integration = scp_sequential_2(robot, X_warm[-1], U_warm[-1], dt, trust_region=True, trust_x=2, trust_u=3, num_iterations=scp_epoch)

  # in roll-out, we always use robot_NN!
  x_rollout, u_rollout, fa = tracking(robot_NN, dt, x0, X_d=X[-1], feedforward=True, ctrl_useNN=False)

  vis_pdf(robot, X_warm[-1], U_warm[-1], X, U, X_integration, X[-1], x_rollout, u_rollout, fa, plot_integration=False, name='PlanwoNN.pdf')
