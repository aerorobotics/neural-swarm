import torch
import math
from nns import rho_Net
from nns import phi_Net


class RobotCrazyFlie2D:
  radius_by_type = {
    'small': 0.15,
    'small_powerful_motors': 0.15,
    'large': 0.2,
  }

  def __init__(self, model_folder, useNN=False, cftype="small"):
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
    self.rho_L_net.load_state_dict(torch.load('{}/rho_L.pth'.format(model_folder)))
    self.phi_L_net.load_state_dict(torch.load('{}/phi_L.pth'.format(model_folder)))
    self.rho_S_net = rho_Net(H=self.H)
    self.phi_S_net = phi_Net(H=self.H)
    self.rho_S_net.load_state_dict(torch.load('{}/rho_S.pth'.format(model_folder)))
    self.phi_S_net.load_state_dict(torch.load('{}/phi_S.pth'.format(model_folder)))
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

  def __init__(self, model_folder, useNN=False, cftype="small"):
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
    self.rho_L_net.load_state_dict(torch.load('{}/rho_L.pth'.format(model_folder)))
    self.phi_L_net.load_state_dict(torch.load('{}/phi_L.pth'.format(model_folder)))
    self.rho_S_net = rho_Net(H=self.H)
    self.phi_S_net = phi_Net(H=self.H)
    self.rho_S_net.load_state_dict(torch.load('{}/rho_S.pth'.format(model_folder)))
    self.phi_S_net.load_state_dict(torch.load('{}/phi_S.pth'.format(model_folder)))
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
        if abs(x_12[0]) < 0.2 and abs(x_12[1]) < 0.2 and abs(x_12[3]) < 1.5:
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