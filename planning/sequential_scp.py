import torch
import cvxpy as cp
import numpy as np

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


def jacobian(robot, xbar, ubar, data_neighbors, dt):
  xbar.requires_grad = True
  ubar.requires_grad = True
  y = robot.step(xbar, ubar, data_neighbors, dt)
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

def scp(robot, initial_x, initial_u, xf, dt, data_neighbors, trust_region=False, trust_x=2, trust_u=2, num_iterations=10):
  X, U = [initial_x], [initial_u]
  X_integration = [None]

  xprev = initial_x
  uprev = initial_u
  T = xprev.shape[0]

  x0 = xprev[0]

  for iteration in range(num_iterations):

    x = cp.Variable((T, robot.stateDim))
    u = cp.Variable((T-1, robot.ctrlDim))

    objective = cp.Minimize(cp.sum_squares(u))# + 1e6 * cp.norm(x[-1,0:4] - xf[0:4], "inf"))

    constraints = [
      x[0] == x0, # initial state constraint
      x[-1,0:4] == initial_x[-1,0:4], # goal state constraint
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

    # dynamics constraints
    for t in range(0, T-1):
      xbar = xprev[t]
      ubar = uprev[t]

      data_neighbors_t = get_data_neighbors(data_neighbors, t)
      A, B, y = jacobian(robot, xbar, ubar, data_neighbors_t, dt)
      # simple version:
      constraints.append(
        x[t+1] == y.numpy() + A.numpy() @ (x[t] - xbar.numpy()) + B.numpy() @ (u[t] - ubar.numpy())
        )

      if robot.useNN:
        # print(data_neighbors_t)
        for k, (cftype_neighbor, x_neighbor) in enumerate(data_neighbors_t):
          other_neighbors = [(robot.cftype, xbar)] + data_neighbors_t[0:k-1] + data_neighbors_t[k+1:]

          # print(x_neighbor, other_neighbors, cftype_neighbor)
          # exit()

          xbar.requires_grad = True
          Fa_neighbor = robot.compute_Fa(x_neighbor, other_neighbors, cftype=cftype_neighbor)
          Fa_neighbor_grad = torch.autograd.grad(outputs=[Fa_neighbor], inputs=[xbar], allow_unused=True)[0]
          xbar.requires_grad = False
          # print(t, k, Fa_neighbor_grad, Fa_neighbor.detach())
     
          if cftype_neighbor == "small":
            limit = 5
            mass = 34
          elif cftype_neighbor == "large":
            limit = 10
            mass = 67
          if abs(Fa_neighbor.detach().numpy()) > limit:
            print("Warning: neighbors Fa is out of bounds at t {}: {}".format(t, Fa_neighbor.detach().numpy()))
          if abs(Fa_neighbor.detach().numpy()*9.81/mass - x_neighbor[4]) > 0.1:
            print("Warning: neighbor Fa estimate different from its state at t {}: {} {}".format(t, Fa_neighbor.detach().numpy(), x_neighbor[4]/9.81*mass))
     
          if Fa_neighbor_grad is not None:
            # print(t,k,cftype_neighbor,Fa_neighbor.detach().numpy(),Fa_neighbor_grad.detach().numpy())

            # ## sanity check
            # xbar2 = xbar + torch.tensor([0.01,0,0,0,0])
            # other_neighbors2 = [(robot.cftype, xbar2)] + data_neighbors_t[0:k-1] + data_neighbors_t[k+1:]
            # Fa_neighbor2 = robot.compute_Fa(x_neighbor, other_neighbors2, cftype=cftype_neighbor)

            # print(Fa_neighbor2, Fa_neighbor.detach().numpy() + Fa_neighbor_grad.detach().numpy() @ (xbar2.numpy() - xbar.numpy()))
            # # exit()
            # ####

            constraints.extend([
              cp.abs((Fa_neighbor.detach().numpy() + Fa_neighbor_grad.detach().numpy() @ (x[t] - xbar.numpy()))*9.81/mass - x_neighbor[4]) <= 0.3,
              cp.abs(Fa_neighbor.detach().numpy() + Fa_neighbor_grad.detach().numpy() @ (x[t] - xbar.numpy())) <= limit,
            ])


      # sanity check of quality of current estimate
      xnext = y
      diff = (xprev[t+1] - xnext).abs()
      if (diff > torch.tensor([0.01,0.01,0.05,0.05,0.3])).any():
        print("Warning: bad linearization at t {}: {}".format(t, diff))
        xprev[t+1,4] = xnext[4]

      # sanity check of bounds
      if (xnext < torch.tensor(robot.x_min)).any() or (xnext > torch.tensor(robot.x_max)).any():
        print("Warning: out of bounds at t {}: {} ".format(t, xnext))

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
        robot.x_min <= x[t],
        x[t] <= robot.x_max
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
      print("Warning: Solver failed!")
      exit()
      return X, U, X_integration, float('inf')
    except KeyError:
      print("Warning BarQCPConvTol too big?")
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
    X.append(xprev)
    U.append(uprev)

    x_int = torch.zeros((T, robot.stateDim))
    x_int[0] = x0
    for t in range(0, T-1):
      data_neighbors_t = get_data_neighbors(data_neighbors, t)
      x_int[t+1] = x_int[t] + dt * (robot.f(x_int[t], uprev[t], data_neighbors_t, dt))
    X_integration.append(x_int.detach())

  return X, U, X_integration, prob.value


def scp_min_xf(robot, initial_x, initial_u, xf, dt, data_neighbors, trust_region=False, trust_x=2, trust_u=2, num_iterations=10):
  X, U = [initial_x], [initial_u]
  X_integration = [None]

  xprev = initial_x
  uprev = initial_u
  T = xprev.shape[0]

  x0 = xprev[0]

  for iteration in range(num_iterations):

    x = cp.Variable((T, robot.stateDim))
    u = cp.Variable((T-1, robot.ctrlDim))
    # delta = cp.Variable()

    objective = cp.Minimize(cp.norm(x[-1,0:4] - xf[0:4], "inf"))

    constraints = [
      x[0] == x0, # initial state constraint
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

    # dynamics constraints
    x_min = xprev[0]
    x_max = xprev[0]
    for t in range(0, T-1):
      xbar = xprev[t]
      ubar = uprev[t]

      # print(other_x)
      data_neighbors_t = get_data_neighbors(data_neighbors, t)
      # print(x_neighbors)
      A, B, y = jacobian(robot, xbar, ubar, data_neighbors_t, dt)
      # simple version:
      constraints.append(
        x[t+1] == x[t] + dt * (y.numpy() + A.numpy() @ (x[t] - xbar.numpy()) + B.numpy() @ (u[t] - ubar.numpy()))
        )

      x_min = np.minimum(x_min, xbar + dt * y.numpy())
      x_max = np.maximum(x_max, xbar + dt * y.numpy())

      if robot.useNN:
        for k, (cftype_neighbor, x_neighbor) in enumerate(data_neighbors_t):
          other_neighbors = [(robot.cftype, xbar)] + data_neighbors_t[0:k-1] + data_neighbors_t[k+1:]

          xbar.requires_grad = True
          Fa_neighbor = robot.compute_Fa(x_neighbor, other_neighbors, cftype=cftype_neighbor)
          Fa_neighbor_grad = torch.autograd.grad(outputs=[Fa_neighbor], inputs=[xbar], allow_unused=True)[0]
          xbar.requires_grad = False
          # print(t, k, Fa_neighbor_grad, Fa_neighbor.detach())
          if Fa_neighbor_grad is not None:
            if cftype_neighbor == "small":
              limit = 5
            elif cftype_neighbor == "large":
              limit = 10
            if abs(Fa_neighbor.detach().numpy()) > limit:
              print("Warning: neighbors Fa is out of bounds at t {}: {}".format(t, Fa_neighbor.detach().numpy()))
            # print(t,k,cftype_neighbor,Fa_neighbor.detach().numpy(),Fa_neighbor_grad.detach().numpy())

            # ## sanity check
            # xbar2 = xbar + torch.tensor([0.01,0,0,0,0])
            # other_neighbors2 = [(robot.cftype, xbar2)] + data_neighbors_t[0:k-1] + data_neighbors_t[k+1:]
            # Fa_neighbor2 = robot.compute_Fa(x_neighbor, other_neighbors2, cftype=cftype_neighbor)

            # print(Fa_neighbor2, Fa_neighbor.detach().numpy() + Fa_neighbor_grad.detach().numpy() @ (xbar2.numpy() - xbar.numpy()))
            # # exit()
            # ####


            constraints.append(
              cp.abs(Fa_neighbor.detach().numpy() + Fa_neighbor_grad.detach().numpy() @ (x[t] - xbar.numpy())) <= limit
            )

      # sanity check of quality of current estimate
      xnext = xprev[t] + dt * y
      diff = (xprev[t+1] - xnext).abs()
      if (diff > 0.1).any():
        print("Warning: bad linearization at t {}: {}".format(t, diff))

      # sanity check of bounds
      if (xnext < torch.tensor(robot.x_min)).any() or (xnext > torch.tensor(robot.x_max)).any():
        print("Warning: out of bounds at t {}: {} ".format(t, xnext))


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
        robot.x_min <= x[t],
        x[t] <= robot.x_max,
        ])

      # constraints.extend([
      #   # x_min <= x[t],                # hard constraint on initial traj limit
      #   # robot.x_min - delta <= x[t],  # soft constraint on physical limit

      #   # x[t] <= x_max, 
      #   # x[t] <= robot.x_max + delta,

      #   # np.minimum(robot.x_min, x_min*0.99) <= x[t],
      #   # x[t] <= np.maximum(robot.x_max, x_max*0.99),
      #   ])

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
      print("Warning: Solver failed!", x_min, x_max)
      exit()
      return X, U, X_integration, float('inf')
    except KeyError:
      print("Warning BarQCPConvTol too big?")
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
    X.append(xprev)
    U.append(uprev)

    x_int = torch.zeros((T, robot.stateDim))
    x_int[0] = x0
    for t in range(0, T-1):
      data_neighbors_t = get_data_neighbors(data_neighbors, t)
      x_int[t+1] = x_int[t] + dt * (robot.f(x_int[t], uprev[t], data_neighbors_t, dt))
    X_integration.append(x_int.detach())

    if prob.value < 1e-8:
      break

  return X, U, X_integration, prob.value
