from nns import phi_Net, rho_Net
from utlis import vis
import torch

phi_L_net = phi_Net(H=20)
phi_S_net = phi_Net(H=20)
rho_L_net = rho_Net(H=20)
rho_S_net = rho_Net(H=20)

rho_L_net.load_state_dict(torch.load('rho_L.pth'))
rho_S_net.load_state_dict(torch.load('rho_S.pth'))
phi_L_net.load_state_dict(torch.load('phi_L.pth'))
phi_S_net.load_state_dict(torch.load('phi_S.pth'))

vis(phi_L_net, rho_L_net, phi_S_net, rho_S_net)