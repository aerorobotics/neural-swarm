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

#### Two Crazyflies simulator ####
from nns import rho_Net
from nns import phi_Net
class RobotTwoCrazyFlies2D:
  def __init__(self, useNN=False):
    # x = [py1, pz1, vy1, vz1, py2, pz2, vy2, vz2]
    # u = [fdy1, fdz1, fdy2, fdz2]
    self.stateDim = 8
    self.ctrlDim = 4
    self.x_min = [-1, -1, -5, -5, -1, -1, -5, -5]
    self.x_max = [1, 1, 5, 5, 1, 1, 5, 5]
    self.thrust_to_weight = 2.6
    self.g = 9.81
    self.mass = 34
    self.radius = 0.2

    self.rho_net = rho_Net()
    self.phi_net = phi_Net()
    self.rho_net.load_state_dict(torch.load('./rho_0912_3.pth'))
    self.phi_net.load_state_dict(torch.load('./phi_0912_3.pth'))
    self.nn = useNN

  def f(self, x, u):
    x_12 = torch.zeros(6)
    x_12[1:3] = x[4:6] - x[:2]
    x_12[4:] = x[6:] - x[2:4] 
    faz_1 = self.rho_net(self.phi_net(x_12)) # unit: gram
    faz_2 = self.rho_net(self.phi_net(-x_12))
    
    weight = 0.0
    if self.nn:
      weight = 1.0
    
    return torch.stack([
      x[2],
      x[3],
      u[0],
      u[1] + self.g*(weight*faz_1[0]/self.mass-1.0),
      x[6],
      x[7],
      u[2],
      u[3] + self.g*(weight*faz_2[0]/self.mass-1.0)])