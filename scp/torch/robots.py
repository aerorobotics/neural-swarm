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
    self.u_min = [-2 * 9.81, -2 * 9.81, -2 * 9.81, -2 * 9.81]
    self.u_max = [2 * 9.81, 2 * 9.81, 2 * 9.81, 2 * 9.81]
    self.thrust_to_weight = 2.6
    self.g = 9.81
    self.mass = 34
    self.radius = 0.2

    self.rho_net = rho_Net(H=40)
    self.phi_net = phi_Net(H=40)
    self.rho_net.load_state_dict(torch.load('./models/rho_0912_3.pth'))
    self.phi_net.load_state_dict(torch.load('./models/phi_0912_3.pth'))
    self.useNN = useNN
    self.dim_deepset = 3 # whether or not we consider velocity as the deepset's input

    # controller
    self.kp = 4.0 # 20.0 # 1.0
    self.kd = 4.0 # 20.0 # 2.0

  def compute_Fa(self, x, useNN_override=None):
    useNN = self.useNN
    if useNN_override is not None:
      useNN = useNN_override

    if useNN:
      x_12 = torch.zeros(6)
      x_12[1:3] = x[4:6] - x[:2]
      if self.dim_deepset == 6:
        x_12[4:] = x[6:] - x[2:4] 
      faz_1 = self.rho_net(self.phi_net(x_12)) # unit: gram
      faz_2 = self.rho_net(self.phi_net(-x_12))

      return torch.stack([faz_1[0], faz_2[0]])
    else:
      return torch.zeros(2)

  def f(self, x, u, useNN=None):

    Fa = self.compute_Fa(x, useNN)
    return torch.stack([
      x[2],
      x[3],
      u[0],
      u[1] + self.g*(Fa[0]/self.mass-1.0),
      x[6],
      x[7],
      u[2],
      u[3] + self.g*(Fa[1]/self.mass-1.0)])

  def controller(self, x, x_d, v_d_dot, useNN=None):
    Fa = self.compute_Fa(x, useNN)

    u_des = torch.stack([
      -self.kp*(x[0]-x_d[0]) - self.kd*(x[2]-x_d[2]) + v_d_dot[0],
      # -self.kp*(x[1]-x_d[1]) - self.kd*(x[3]-x_d[3]) - self.g*(Fa[0]/self.mass-1.0) + v_d_dot[1],
      -self.kp*(x[1]-x_d[1]) - self.kd*(x[3]-x_d[3])  + v_d_dot[1],
      -self.kp*(x[4]-x_d[4]) - self.kd*(x[6]-x_d[6]) + v_d_dot[2],
      # -self.kp*(x[5]-x_d[5]) - self.kd*(x[7]-x_d[7]) - self.g*(Fa[1]/self.mass-1.0) + v_d_dot[3]])
      -self.kp*(x[5]-x_d[5]) - self.kd*(x[7]-x_d[7]) + v_d_dot[3]])

    # u_des = v_d_dot
    # return u_des

    # if the controller outputs a desired value above our limit, scale the vector (keeping its direction)
    u_des_norm = u_des.norm()
    if u_des_norm > self.g * self.thrust_to_weight:
      return u_des / u_des_norm * self.g * self.thrust_to_weight
    else:
      return u_des


class RobotCrazyFlie2D:
  radius_by_type = {
    'small': 0.15,
    'small_powerful_motors': 0.15,
    'large': 0.2,
  }

  def __init__(self, useNN=False, cftype="small"):
    # x = [py, pz, vy, vz]
    # u = [fdy, fdz]
    self.stateDim = 4
    self.ctrlDim = 2
    self.x_min = [-1, -1, -5, -5]
    self.x_max = [1, 1, 5, 5]
    self.g = 9.81
    self.cftype = cftype
    self.radius = RobotCrazyFlie2D.radius_by_type[self.cftype]
    if self.cftype == "small":
      self.thrust_to_weight = 1.4 # default motors: 1.4; upgraded motors: 2.6
      self.mass = 34 # g
    elif self.cftype == "small_powerful_motors":
      self.thrust_to_weight = 2.6 # default motors: 1.4; upgraded motors: 2.6
      self.mass = 34 # g
    elif self.cftype == "large":
      self.thrust_to_weight = 2.1 # max thrust: ~145g; 
      self.mass = 67 # g
    else:
      raise Exception("Unknown cftype!")

    self.H = 20 # dimension of the hidden state (output of \phi nets)
    self.rho_L_net = rho_Net(H=self.H)
    self.phi_L_net = phi_Net(H=self.H)
    self.rho_L_net.load_state_dict(torch.load('./models/rho_L.pth'))
    self.phi_L_net.load_state_dict(torch.load('./models/phi_L.pth'))
    self.rho_S_net = rho_Net(H=self.H)
    self.phi_S_net = phi_Net(H=self.H)
    self.rho_S_net.load_state_dict(torch.load('./models/rho_S.pth'))
    self.phi_S_net.load_state_dict(torch.load('./models/phi_S.pth'))
    self.useNN = useNN
    self.dim_deepset = 3 # whether or not we consider velocity as the deepset's input

    # controller
    self.kp = 4.0 # 1.0
    self.kd = 4.0 # 2.0

  def min_distance(self, cftype_neighbor):
    return self.radius + RobotCrazyFlie2D.radius_by_type[cftype_neighbor]

  def compute_Fa(self, x, data_neighbors, useNN_override=None):
    useNN = self.useNN
    if useNN_override is not None:
      useNN = useNN_override

    if useNN:
      rho_input = torch.zeros(self.H)
      for cftype_neighbor, x_neighbor in data_neighbors:
        x_12 = torch.zeros(6)
        x_12[1:3] = x_neighbor[0:2] - x[0:2]
        if self.dim_deepset == 6:
          x_12[4:] = x_neighbor[2:4] - x[2:4]
        if cftype_neighbor == "small" or cftype_neighbor == "small_powerful_motors":
          rho_input += self.phi_S_net(x_12)
        elif cftype_neighbor == "large":
          rho_input += self.phi_L_net(x_12)
        else:
          raise Exception("Unknown cftype!")

      if self.cftype == "small" or self.cftype == "small_powerful_motors":
        faz = self.rho_S_net(rho_input)
      elif self.cftype == "large":
        faz = self.rho_L_net(rho_input)
      else:
        raise Exception("Unknown cftype!")

      return faz[0]
    else:
      return 0.0

  def f(self, x, u, data_neighbors, useNN=None):

    Fa = self.compute_Fa(x, data_neighbors, useNN)
    return torch.stack([
      x[2],
      x[3],
      u[0],
      u[1] + self.g*(Fa/self.mass-1.0)])

  def controller(self, x, x_d, v_d_dot):
    u_des = torch.stack([
      -self.kp*(x[0]-x_d[0]) - self.kd*(x[2]-x_d[2]) + v_d_dot[0],
      -self.kp*(x[1]-x_d[1]) - self.kd*(x[3]-x_d[3]) + v_d_dot[1]])

    # u_des = v_d_dot
    # return u_des

    # if the controller outputs a desired value above our limit, scale the vector (keeping its direction)
    u_des_norm = u_des.norm()
    if u_des_norm > self.g * self.thrust_to_weight:
      return u_des / u_des_norm * self.g * self.thrust_to_weight
    else:
      return u_des

# TODO: might be nicer to use a disk-shape collision model?
class RobotCrazyFlie3D:
  radius_by_type = {
    'small': 0.15,
    'small_powerful_motors': 0.15,
    'large': 0.2,
  }

  def __init__(self, useNN=False, cftype="small"):
    # x = [px, py, pz, px, vy, vz]
    # u = [fdx, fdy, fdz]
    self.stateDim = 6
    self.ctrlDim = 3
    self.x_min = [-1, -1, -1, -5, -5, -5]
    self.x_max = [ 1,  1,  1,  5,  5,  5]
    self.g = 9.81
    self.cftype = cftype
    self.radius = RobotCrazyFlie2D.radius_by_type[self.cftype]
    if self.cftype == "small":
      self.thrust_to_weight = 1.4 # default motors: 1.4; upgraded motors: 2.6
      self.mass = 34 # g
    elif self.cftype == "small_powerful_motors":
      self.thrust_to_weight = 2.6 # default motors: 1.4; upgraded motors: 2.6
      self.mass = 34 # g
    elif self.cftype == "large":
      self.thrust_to_weight = 2.1 # max thrust: ~145g; 
      self.mass = 67 # g
    else:
      raise Exception("Unknown cftype!")

    self.H = 20 # dimension of the hidden state (output of \phi nets)
    self.rho_L_net = rho_Net(H=self.H)
    self.phi_L_net = phi_Net(H=self.H)
    self.rho_L_net.load_state_dict(torch.load('./models/rho_L.pth'))
    self.phi_L_net.load_state_dict(torch.load('./models/phi_L.pth'))
    self.rho_S_net = rho_Net(H=self.H)
    self.phi_S_net = phi_Net(H=self.H)
    self.rho_S_net.load_state_dict(torch.load('./models/rho_S.pth'))
    self.phi_S_net.load_state_dict(torch.load('./models/phi_S.pth'))
    self.useNN = useNN

    # controller
    self.kp = 4.0 # 1.0
    self.kd = 4.0 # 2.0

  def min_distance(self, cftype_neighbor):
    return self.radius + RobotCrazyFlie2D.radius_by_type[cftype_neighbor]

  def compute_Fa(self, x, data_neighbors, useNN_override=None):
    useNN = self.useNN
    if useNN_override is not None:
      useNN = useNN_override

    if useNN:
      rho_input = torch.zeros(self.H)
      for cftype_neighbor, x_neighbor in data_neighbors:
        x_12 = torch.zeros(6)
        x_12 = (x_neighbor - x).float()
        if cftype_neighbor == "small" or cftype_neighbor == "small_powerful_motors":
          rho_input += self.phi_S_net(x_12)
        elif cftype_neighbor == "large":
          rho_input += self.phi_L_net(x_12)
        else:
          raise Exception("Unknown cftype!")

      if self.cftype == "small" or self.cftype == "small_powerful_motors":
        faz = self.rho_S_net(rho_input)
      elif self.cftype == "large":
        faz = self.rho_L_net(rho_input)
      else:
        raise Exception("Unknown cftype!")

      return faz[0]
    else:
      return 0.0

  def f(self, x, u, data_neighbors, useNN=None):

    Fa = self.compute_Fa(x, data_neighbors, useNN)
    return torch.stack([
      x[3],
      x[4],
      x[5],
      u[0],
      u[1],
      u[2] + self.g*(Fa/self.mass-1.0)])

  def controller(self, x, x_d, v_d_dot):
    u_des = torch.stack([
      -self.kp*(x[0]-x_d[0]) - self.kd*(x[3]-x_d[3]) + v_d_dot[0],
      -self.kp*(x[1]-x_d[1]) - self.kd*(x[4]-x_d[4]) + v_d_dot[1],
      -self.kp*(x[2]-x_d[2]) - self.kd*(x[5]-x_d[5]) + v_d_dot[2]])

    # u_des = v_d_dot
    # return u_des

    # if the controller outputs a desired value above our limit, scale the vector (keeping its direction)
    u_des_norm = u_des.norm()
    if u_des_norm > self.g * self.thrust_to_weight:
      return u_des / u_des_norm * self.g * self.thrust_to_weight
    else:
      return u_des