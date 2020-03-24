import torch
import math

class RobotDubinsCar:
  def __init__(self, v, k):
    self.v = v
    self.k = k

    self.stateDim = 3
    self.ctrlDim = 1
    self.x_min = [0, 0, -math.pi]
    self.x_max = [60, 60, math.pi]
    self.u_min = [-2]
    self.u_max = [2]

  def f(self, x, u):
    return torch.stack((
      self.v * torch.cos(x[2]),
      self.v * torch.sin(x[2]),
      self.k * u[0]))


class RobotDoubleIntegrator:
  def __init__(self):
    self.stateDim = 4
    self.ctrlDim = 2
    self.x_min = [-6, -6, -10, -10]
    self.x_max = [6, 6, 10, 10]
    self.u_min = [-2, -2]
    self.u_max = [2, 2]

  def f(self, x, u):
    return torch.stack([
      x[2],
      x[3],
      u[0],
      u[1]])


class RobotAirplane:
  def __init__(self):
    self.mass = 1
    self.rho = 1.225
    self.area = 0.7
    self.Cd0 = 0.015
    self.Kd = 0.025
    self.v_min = 3
    self.v_max = 10
    self.g = 9.81

    self.stateDim = 8
    self.ctrlDim = 3

    self.x_min = [-11, -11, -11, -3.14159, 3, -0.523599, -0.785398, -3.14159]
    self.x_max = [11, 11, 11, 3.14159, 10, 0.523599, 0.785398, 3.14159]
    self.u_min = [-8.05686, -2.0944, -1.0472]
    self.u_max = [8.05686, 2.0944, 1.0472]

  # x dot = f(x, u)
  def f(self, x, u):
    _,_,_,psi,v,gamma,phi,alpha = x
    Fl = math.pi * self.rho * self.area * v * v * alpha
    Fd = self.rho*self.area * v *v * (self.Cd0 + 4 * math.pi * math.pi * self.Kd * alpha * alpha)
    return torch.stack((
      v * torch.cos(psi) * torch.cos(gamma),
      v * torch.sin(psi) * torch.cos(gamma),
      v * torch.sin(gamma),
      -Fl * torch.sin(phi) / (self.mass * v * torch.cos(gamma)),
      u[0] - Fd/self.mass - self.g * torch.sin(gamma),
      Fl * torch.cos(phi)/(self.mass*v) - self.g * torch.cos(gamma)/v,
      u[1],
      u[2]))