import torch
import math
import logging
from nns import rho_Net
from nns import phi_Net


class RobotCrazyFlie2D:
  properties_by_type = {
    'small': {
      'radius': 0.1,
      'mass': 34, #g
      'thrust_to_weight': 1.4,
      # 'max_Fa': 5, #g
      'max_Fa': 5, #g
      'trust_Fa': 5, #g
    },
    'small_powerful_motors': {
      'radius': 0.1,
      'mass': 34, #g
      'thrust_to_weight': 2.6,
      # 'max_Fa': 8, #g
      'max_Fa': 8, #g
      'trust_Fa': 5, #g
    },
    'large': {
      'radius': 0.15,
      'mass': 67, #g
      'thrust_to_weight': 2.1,
      # 'max_Fa': 10, #g
      'max_Fa': 10, #g
      'trust_Fa': 5, #g
    }
  }

  def __init__(self, model_folder, useNN=False, cftype="small", xy_filter=math.inf):
    # x = [py, pz, vy, vz, faz]
    # u = [fdy, fdz]
    self.stateDim = 5
    self.ctrlDim = 2
    self.x_min = [-1, 0, -0.5, -0.5, -5]
    self.x_max = [1, 2, 0.5, 0.5, 5]
    self.g = 9.81
    self.cftype = cftype
    self.radius = RobotCrazyFlie2D.properties_by_type[self.cftype]['radius']
    self.mass = RobotCrazyFlie2D.properties_by_type[self.cftype]['mass']
    self.thrust_to_weight = RobotCrazyFlie2D.properties_by_type[self.cftype]['thrust_to_weight']
    max_Fa = RobotCrazyFlie2D.properties_by_type[self.cftype]['max_Fa']
    self.x_min[-1] = -max_Fa
    self.x_max[-1] = max_Fa

    self.H = 20
    self.rho_L_net = rho_Net(hiddendim=self.H)
    self.phi_L_net = phi_Net(inputdim=6,hiddendim=self.H) #x,y,z,vx,vy,vz
    self.rho_L_net.load_state_dict(torch.load('{}/rho_L.pth'.format(model_folder)))
    self.phi_L_net.load_state_dict(torch.load('{}/phi_L.pth'.format(model_folder)))
    self.rho_S_net = rho_Net(hiddendim=self.H)
    self.phi_S_net = phi_Net(inputdim=6,hiddendim=self.H) #x,y,z,vx,vy,vz
    self.rho_S_net.load_state_dict(torch.load('{}/rho_S.pth'.format(model_folder)))
    self.phi_S_net.load_state_dict(torch.load('{}/phi_S.pth'.format(model_folder)))
    self.phi_G_net = phi_Net(inputdim=4,hiddendim=self.H) #z,vx,vy,vz
    self.phi_G_net.load_state_dict(torch.load('{}/phi_G.pth'.format(model_folder)))
    self.useNN = useNN
    self.use_relative_velocity = True # whether or not we consider velocity as the deepset's input
    self.use_ground = True
    self.xy_filter = xy_filter

    # controller
    self.kp = 8
    self.kd = 5
    self.ki = 25.0
    self.i_part = torch.zeros(2)
    self.i_limit = 1e-1


  def min_distance(self, cftype_neighbor):
    return self.radius + RobotCrazyFlie2D.properties_by_type[cftype_neighbor]['radius']

  def max_Fa(self, cftype):
    return RobotCrazyFlie2D.properties_by_type[cftype]['max_Fa']

  def trust_Fa(self, cftype):
    return RobotCrazyFlie2D.properties_by_type[cftype]['trust_Fa']

  # def compute_Fa(self, x, data_neighbors, useNN_override=None, cftype=None):
  #   useNN = self.useNN
  #   if useNN_override is not None:
  #     useNN = useNN_override

  #   if cftype is None:
  #     cftype = self.cftype

  #   if useNN:
  #     output = torch.zeros(1,dtype=torch.float32)
  #     for cftype_neighbor, x_neighbor in data_neighbors:
  #       rel = x_neighbor[0:2] - x[0:2]
  #       if rel[1] > 0:
  #         output += torch.exp(-torch.pow(rel[0]/0.1, 2)) * -3/(torch.abs(rel[1]+0.01))
  #       else:
  #         output += torch.exp(-torch.pow(rel[0]/0.1, 2)) * -0.5/(torch.abs(rel[1]+0.01))

  #     return output[0]
  #   else:
  #     return torch.zeros(1,dtype=torch.float32)[0]

  def compute_Fa(self, x, data_neighbors, useNN_override=None, cftype=None):
    useNN = self.useNN
    if useNN_override is not None:
      useNN = useNN_override

    if cftype is None:
      cftype = self.cftype

    if useNN:
      # if x[0] > -0.3 and x[0] < -0.2:
      #   return -12.0
      # return 2.0

      rho_input = torch.zeros(self.H, dtype=torch.float32)
      for cftype_neighbor, x_neighbor in data_neighbors:
        if abs(x_neighbor[0] - x[0]) <= self.xy_filter:
          x_12 = torch.zeros(6, dtype=torch.float32)
          x_12[1:3] = x_neighbor[0:2] - x[0:2]
          if self.use_relative_velocity:
            x_12[4:] = x_neighbor[2:4] - x[2:4]
          if cftype_neighbor == "small" or cftype_neighbor == "small_powerful_motors":
            rho_input += self.phi_S_net(x_12)
          elif cftype_neighbor == "large":
            rho_input += self.phi_L_net(x_12)
          else:
            raise Exception("Unknown cftype!")

      # interaction with the ground
      if self.use_ground:
        x_12 = torch.zeros(4, dtype=torch.float32)
        x_12[0] = self.x_min[1] - x[1]
        if self.use_relative_velocity:
          x_12[2:4] = -x[2:4]
        rho_input += self.phi_G_net(x_12)

      if cftype == "small" or cftype == "small_powerful_motors":
        faz = self.rho_S_net(rho_input)
      elif cftype == "large":
        faz = self.rho_L_net(rho_input)
      else:
        raise Exception("Unknown cftype!")

      return faz[0]
    else:
      return torch.zeros(1,dtype=torch.float32)[0]

  # def f(self, x, u, data_neighbors, useNN=None, dt=0.05):

  #   Fa = self.compute_Fa(x, data_neighbors, useNN)
  #   return torch.stack([
  #     x[2],
  #     x[3],
  #     u[0],
  #     u[1] + self.g*(Fa/self.mass-1.0),
  #     (self.g*Fa/self.mass - x[4]) / dt]) # TODO: this is a bit hacky...

  def step(self, x, u, data_neighbors_next, dt, useNN=None):
    current_Fa = x[4]
    next_x = torch.stack([
      x[0] + dt * x[2],
      x[1] + dt * x[3],
      x[2] + dt * u[0],
      x[3] + dt * (u[1] + self.g*(current_Fa/self.mass-1.0))])
    next_Fa = self.compute_Fa(next_x, data_neighbors_next, useNN)
    return torch.stack((next_x[0], next_x[1], next_x[2], next_x[3], next_Fa))

  def controller_reset(self):
    self.i_part = torch.zeros(2)

  def controller(self, x, x_d, v_d_dot, dt):
    self.i_part += (x[0:2]-x_d[0:2]) * dt
    self.i_part = torch.clamp(self.i_part, -self.i_limit, self.i_limit)
    # print(self.i_part)
    u_des = torch.stack([
      -self.kp*(x[0]-x_d[0]) - self.kd*(x[2]-x_d[2]) - self.ki*self.i_part[0] + v_d_dot[0],
      -self.kp*(x[1]-x_d[1]) - self.kd*(x[3]-x_d[3]) - self.ki*self.i_part[1] + v_d_dot[1]])

    # u_des = v_d_dot
    # return u_des

    # if the controller outputs a desired value above our limit, scale the vector (keeping its direction)
    u_des_norm = u_des.norm()
    if u_des_norm > self.g * self.thrust_to_weight:
      logging.warning("CLAMPING {} {} {} {}".format(u_des_norm, x, x_d, self.i_part))
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
    self.x_min = [-1, -1, -1, -0.5, -0.5, -0.5]
    self.x_max = [ 1,  1,  1,  0.5,  0.5,  0.5]
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

    self.H = 20
    self.rho_L_net = rho_Net(hiddendim=self.H)
    self.phi_L_net = phi_Net(inputdim=6,hiddendim=self.H) #x,y,z,vx,vy,vz
    self.rho_L_net.load_state_dict(torch.load('{}/rho_L.pth'.format(model_folder)))
    self.phi_L_net.load_state_dict(torch.load('{}/phi_L.pth'.format(model_folder)))
    self.rho_S_net = rho_Net(hiddendim=self.H)
    self.phi_S_net = phi_Net(inputdim=6,hiddendim=self.H) #x,y,z,vx,vy,vz
    self.rho_S_net.load_state_dict(torch.load('{}/rho_S.pth'.format(model_folder)))
    self.phi_S_net.load_state_dict(torch.load('{}/phi_S.pth'.format(model_folder)))
    self.phi_G_net = phi_Net(inputdim=4,hiddendim=self.H) #z,vx,vy,vz
    self.phi_G_net.load_state_dict(torch.load('{}/phi_G.pth'.format(model_folder)))
    self.useNN = useNN

    # controller
    self.kp = 16
    self.kd = 10
    self.ki = 50.0
    self.i_part = torch.zeros(2)
    self.i_limit = 1e-1

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

      # interaction with the ground
      x_12 = torch.zeros(4)
      x_12[0] = self.x_min[2] - x[2]
      x_12[1:4] = -x[3:6]
      rho_input += self.phi_G_net(x_12)

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

  def controller(self, x, x_d, v_d_dot, dt):
    self.i_part += (x[0:3]-x_d[0:3]) * dt
    self.i_part = torch.clamp(self.i_part, -self.i_limit, self.i_limit)
    u_des = torch.stack([
      -self.kp*(x[0]-x_d[0]) - self.kd*(x[3]-x_d[3]) -self.ki*self.i_part[0] + v_d_dot[0],
      -self.kp*(x[1]-x_d[1]) - self.kd*(x[4]-x_d[4]) -self.ki*self.i_part[1] + v_d_dot[1],
      -self.kp*(x[2]-x_d[2]) - self.kd*(x[5]-x_d[5]) -self.ki*self.i_part[2] + v_d_dot[2]])

    # u_des = v_d_dot
    # return u_des

    # if the controller outputs a desired value above our limit, scale the vector (keeping its direction)
    u_des_norm = u_des.norm()
    if u_des_norm > self.g * self.thrust_to_weight:
      return u_des / u_des_norm * self.g * self.thrust_to_weight
    else:
      return u_des