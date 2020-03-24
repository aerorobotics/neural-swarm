from robots import RobotDubinsCar, RobotDoubleIntegrator, RobotAirplane
import torch
import math
from scp import scp

import matplotlib.pyplot as plt

def linspace(start, stop, steps):
  return torch.stack([torch.linspace(start[i], stop[i], steps) for i in range(start.size(0))],dim=1)

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

  robot = RobotDoubleIntegrator()

  x0 = torch.tensor([0,0,0,0], dtype=torch.float32)
  xf = torch.tensor([1,2,0,0], dtype=torch.float32)
  dt = 0.1
  T = 10
  num_steps = int(T / dt)

  initial_x = linspace(x0, xf, num_steps)
  initial_u = torch.zeros((num_steps, robot.ctrlDim))

  x, u = scp(robot, initial_x, initial_u, dt, num_iterations=2)

  fig, ax = plt.subplots()
  ax.plot(initial_x[:,0], initial_x[:,1], label="input state space")
  ax.plot(x[:,0], x[:,1], label="opt state space")
  # ax.plot(xprop[:,0], xprop[:,1], label="opt forward prop")

  plt.legend()
  plt.show()

  fig, ax = plt.subplots()
  ax.plot(initial_u[:,0], label="input u (x)")
  ax.plot(initial_u[:,1], label="input u (y)")
  ax.plot(u[:,0], label="opt u (x)")
  ax.plot(u[:,1], label="opt u (y)")
  # ax.plot(xprop[:,0], xprop[:,1], label="opt forward prop")

  plt.legend()
  plt.show()


  fig, ax = plt.subplots()
  ax.plot(initial_x[:,2], label="input vel (x)")
  ax.plot(initial_x[:,3], label="input vel (y)")
  ax.plot(x[:,2], label="opt vel (x)")
  ax.plot(x[:,3], label="opt vel (y)")
  # ax.plot(xprop[:,0], xprop[:,1], label="opt forward prop")

  plt.legend()
  plt.show()