import torch
import cvxpy as cp
import numpy as np
import logging

# Function to compute the jacobian matrices A and B
# note that this is not as straightforward in pyTorch
# see https://discuss.pytorch.org/t/how-to-penalize-norm-of-end-to-end-jacobian/62771/4
# for a detailed discussion
def unit_vectors(length):
  result = []
  for i in range(0, length):
    x = torch.zeros(length)
    x[i] = 1
    result.append(x)
  return result

# def jacobian(robot, xbar, ubar, data_neighbors, dt):
#   xbar.requires_grad = True
#   ubar.requires_grad = True
#   y = robot.f(xbar, ubar, data_neighbors, dt=dt)
#   result = [torch.autograd.grad(outputs=[y], inputs=[xbar], grad_outputs=[unit], retain_graph=True, allow_unused=True)[0] for unit in unit_vectors(y.size(0))]
#   jacobianA = torch.stack(result, dim=0)
#   result = [torch.autograd.grad(outputs=[y], inputs=[ubar], grad_outputs=[unit], retain_graph=True, allow_unused=True)[0] for unit in unit_vectors(y.size(0))]
#   jacobianB = torch.stack(result, dim=0)

#   xbar.requires_grad = False
#   ubar.requires_grad = False

#   return jacobianA, jacobianB, y.detach()


def jacobian(robot, xbar, ubar, data_neighbors_next, dt):
  xbar.requires_grad = True
  ubar.requires_grad = True
  y = robot.step(xbar, ubar, data_neighbors_next, dt)
  result = [torch.autograd.grad(outputs=[y], inputs=[xbar], grad_outputs=[unit], retain_graph=True, allow_unused=True)[0] for unit in unit_vectors(y.size(0))]
  jacobianA = torch.stack(result, dim=0)
  result = [torch.autograd.grad(outputs=[y], inputs=[ubar], grad_outputs=[unit], retain_graph=True, allow_unused=True)[0] for unit in unit_vectors(y.size(0))]
  jacobianB = torch.stack(result, dim=0)

  xbar.requires_grad = False
  ubar.requires_grad = False

  return jacobianA, jacobianB, y.detach()


def get_data_neighbors(data_neighbors, t):
  data_neighbors_t = []
  for cftype_neighbor, x_neighbor in data_neighbors:
    if t < x_neighbor.shape[0]:
      data_neighbors_t.append((cftype_neighbor, x_neighbor[t].detach()))
    else:
      data_neighbors_t.append((cftype_neighbor, x_neighbor[-1].detach()))
  return data_neighbors_t

def consistency_check(robot, x, u, dt, data_neighbors, trust_fa=2):
  logging.info("Running consistency check...")
  T = x.shape[0]

  for t in range(0, T):
    # check if Fa is correct
    data_neighbors_t = get_data_neighbors(data_neighbors, t)
    Fa = robot.compute_Fa(x[t], data_neighbors_t)
    if abs(Fa.detach().numpy() - x[t,4].numpy()) > robot.trust_Fa(robot.cftype):
      logging.warning("Fa estimate different from its state at t {}: predicted: {:.2f} state: {:.2f}".format(t, Fa.detach().numpy(), x[t,4]))

    # check if there are any collisions
    for cftype_neighbor, xn in data_neighbors_t:
      dist = np.linalg.norm([x[t,0]-xn[0], x[t,1]-xn[1]])
      min_dist = robot.min_distance(cftype_neighbor)
      if dist < min_dist:
        logging.warning("Collision with neighbor at t {}: is: {:.2f} limit: {:.2f}".format(t, dist, min_dist))

    # check propagate state
    if t < T - 1:
      data_neighbors_next = get_data_neighbors(data_neighbors, t+1)
      x_next = robot.step(x[t], u[t], data_neighbors_next, dt)

      diff = (x[t+1] - x_next).abs()
      if (diff > torch.tensor([0.01,0.01,0.05,0.05,robot.trust_Fa(robot.cftype)])).any():
        logging.warning("bad state propagation at t {}: {}".format(t, diff))

    # check state bounds
    if (x[t] < torch.tensor(robot.x_min)).any() or (x[t] > torch.tensor(robot.x_max)).any():
        logging.warning("state out of bounds at t {}: {}".format(t, x[t]))

    # check control bounds
    if t < T - 1:
      if torch.norm(u[t]) > robot.g*robot.thrust_to_weight:
          logging.warning("control out of bounds at t {}: {}".format(t, u[t]))

  logging.info("consistency check done")


def scp(robot, initial_x, initial_u, xf, dt, data_neighbors, trust_region=False, trust_x=2, trust_u=2, num_iterations=10):

  consistency_check(robot, initial_x, initial_u, dt, data_neighbors)

  X, U = [initial_x], [initial_u]
  X_integration = [None]

  xprev = initial_x
  uprev = initial_u
  T = xprev.shape[0]

  x0 = xprev[0]

  for iteration in range(num_iterations):

    x = cp.Variable((T, robot.stateDim))
    u = cp.Variable((T-1, robot.ctrlDim))
    delta = cp.Variable()

    # objective = cp.Minimize(cp.sum_squares(u))
    objective = cp.Minimize(cp.sum_squares(u) + 1e6 * cp.norm(x[-1,0:4] - xf[0:4], "inf") + 1e6 * delta)

    constraints = [
      x[0] == x0, # initial state constraint
      # x[-1,0:4] == initial_x[-1,0:4], # goal state constraint
      delta >= 0,
      # delta == 0,
    ]

    # trust region
    if trust_region:
      for t in range(0, T):
        constraints.append(
          cp.abs(x[t] - xprev[t]) <= trust_x
        )
      for t in range(0, T-1):
        constraints.append(
          cp.abs(u[t] - uprev[t]) <= trust_u
        )
    # correct Fa to reduce linearization errors
    if robot.useNN:
      for t in range(0, T):
        data_neighbors_t = get_data_neighbors(data_neighbors, t)
        Fa = robot.compute_Fa(xprev[t], data_neighbors_t)
        # if abs(Fa.detach().numpy() - xprev[t,4].numpy()) > 1.0:
          # print("Warning: Fa estimate different from its state at t {}: predicted: {:.2f} state: {:.2f}".format(t, Fa.detach().numpy(), xprev[t,4]))
        xprev[t,4] = Fa.detach()

    # dynamics constraints
    for t in range(0, T-1):
      xbar = xprev[t]
      ubar = uprev[t]

      if robot.useNN:
        data_neighbors_t = get_data_neighbors(data_neighbors, t)

        for k, (cftype_neighbor, x_neighbor) in enumerate(data_neighbors_t):
          other_neighbors = [(robot.cftype, xbar)] + data_neighbors_t[0:k] + data_neighbors_t[k+1:]

          # print(x_neighbor, other_neighbors, cftype_neighbor)
          # exit()

          xbar.requires_grad = True
          Fa_neighbor = robot.compute_Fa(x_neighbor, other_neighbors, cftype=cftype_neighbor)
          Fa_neighbor_grad = torch.autograd.grad(outputs=[Fa_neighbor], inputs=[xbar], allow_unused=True)[0]
          xbar.requires_grad = False
          # print(t, k, Fa_neighbor_grad, Fa_neighbor.detach())

          if abs(Fa_neighbor.detach().numpy()) > robot.max_Fa(cftype_neighbor):
            logging.warning("neighbors Fa is out of bounds at t {}: predicted: {:.2f} state: {:.2f}".format(t, Fa_neighbor.detach().numpy(), x_neighbor[4]))
          if abs(Fa_neighbor.detach().numpy() - x_neighbor[4].numpy()) > robot.trust_Fa(cftype_neighbor):
            logging.warning("neighbor Fa estimate different from its state at t {}: predicted: {:.2f} state: {:.2f}".format(t, Fa_neighbor.detach().numpy(), x_neighbor[4]))
     
          if Fa_neighbor_grad is not None:
            # print(t,k,cftype_neighbor,Fa_neighbor.detach().numpy(),Fa_neighbor_grad.detach().numpy())

            # ## sanity check
            # xbar2 = xbar + torch.tensor([0.01,0,0,0,0])
            # other_neighbors2 = [(robot.cftype, xbar2)] + data_neighbors_t[0:k-1] + data_neighbors_t[k+1:]
            # Fa_neighbor2 = robot.compute_Fa(x_neighbor, other_neighbors2, cftype=cftype_neighbor)

            # print(Fa_neighbor2, Fa_neighbor.detach().numpy() + Fa_neighbor_grad.detach().numpy() @ (xbar2.numpy() - xbar.numpy()))
            # exit()
            # ####

            constraints.extend([
              # cp.abs((Fa_neighbor.detach().numpy() + Fa_neighbor_grad.detach().numpy() @ (x[t] - xbar.numpy())) - x_neighbor[4]) <= 2.0,#limit/5,
              cp.abs(Fa_neighbor_grad.detach().numpy() @ (x[t] - xbar.numpy())) <= robot.trust_Fa(cftype_neighbor),
              # cp.abs(Fa_neighbor.detach().numpy() + Fa_neighbor_grad.detach().numpy() @ (x[t] - xbar.numpy())) <= limit,
            ])

      # we need to get the neighbors for the next timestep
      data_neighbors_t = get_data_neighbors(data_neighbors, t+1)
      A, B, y = jacobian(robot, xbar, ubar, data_neighbors_t, dt)
      # simple version:
      constraints.append(
        x[t+1] == y.numpy() + A.numpy() @ (x[t] - xbar.numpy()) + B.numpy() @ (u[t] - ubar.numpy())
        )

      # # sanity check of quality of current estimate
      # xnext = y
      # diff = (xprev[t+1] - xnext).abs()
      # if (diff > torch.tensor([0.01,0.01,0.05,0.05,2.0])).any():
      #   print("Warning: bad linearization at t {}: {}".format(t, diff))
      #   # xprev[t+1,4] = xnext[4]

      # # sanity check of bounds
      # if (xnext < torch.tensor(robot.x_min)).any() or (xnext > torch.tensor(robot.x_max)).any():
      #   print("Warning: propagated out of bounds at t {}: propagated: {} state: {}".format(t, xnext, xprev[t+1]))


      # # discretized zero-order hold
      # F = scipy.linalg.expm(A * dt)
      # G = np.zeros(B.shape)
      # H = np.zeros(xbar.shape)
      # for tau in np.linspace(0, dt, 10):
      #   G += (scipy.linalg.expm(A * tau) @ B) * dt / 10
      #   H += (scipy.linalg.expm(A * tau) @ (robot.f(xbar, ubar) - A @ xbar - B @ ubar)) * dt / 10
      # constraints.append(
      #   x[t+1] == F @ x[t] + G @ u[t] + H
      #   )

    # bounds (x and u)
    for t in range(0, T):
      constraints.extend([
        robot.x_min - delta <= x[t],
        x[t] <= robot.x_max + delta
        ])

      # collision check
      data_neighbors_t = get_data_neighbors(data_neighbors, t)
      for cftype_neighbor, xn in data_neighbors_t:
        xbar = xprev[t]
        dist = np.linalg.norm([xbar[0]-xn[0], xbar[1]-xn[1]])
        min_dist = robot.min_distance(cftype_neighbor)
        constraints.extend([
        (x[t, 0]-xn[0])*(xbar[0]-xn[0]) + (x[t, 1]-xn[1])*(xbar[1]-xn[1]) >= min_dist*dist,
        ])

    for t in range(0, T-1):
      constraints.extend([
        u[t, 0]**2 + u[t, 1]**2 <= (robot.g*robot.thrust_to_weight)**2,
        ])

    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    try:
      result = prob.solve(verbose=False, solver=cp.GUROBI, BarQCPConvTol=1e-9)
    except cp.error.SolverError:
      logging.warning("Solver failed!")
      # exit()
      return X, U, X_integration, float('inf')
    except KeyError:
      logging.warning("BarQCPConvTol too big?")
      return X, U, X_integration, float('inf')

    # dbgx = torch.tensor(x.value, dtype=torch.float32)
    # dbgu = torch.tensor(u.value, dtype=torch.float32)
    # for t in range(0, T):
    #   print(t, dbgx[t] - xprev[t])

    # for t in range(0, T-1):
    #   print(t, dbgu - uprev[t])

    # exit()

    xprev = torch.tensor(x.value, dtype=torch.float32)
    uprev = torch.tensor(u.value, dtype=torch.float32)

    consistency_check(robot, xprev, uprev, dt, data_neighbors)

    X.append(xprev)
    U.append(uprev)

    # x_int = torch.zeros((T, robot.stateDim))
    # x_int[0] = x0
    # for t in range(0, T-1):
    #   data_neighbors_t = get_data_neighbors(data_neighbors, t)
    #   x_int[t+1] = x_int[t] + dt * (robot.f(x_int[t], uprev[t], data_neighbors_t, dt))
    # X_integration.append(x_int.detach())

  return X, U, None, prob.value
