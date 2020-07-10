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

def jacobian(robot, xbar, ubar, data_neighbors):
  xbar.requires_grad = True
  ubar.requires_grad = True
  y = robot.f(xbar, ubar, data_neighbors)
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

def scp(robot, initial_x, initial_u, dt, data_neighbors, trust_region=False, trust_x=2, trust_u=2, num_iterations=10):
  X, U = [initial_x], [initial_u]
  X_integration = [None]

  xprev = initial_x
  uprev = initial_u
  T = xprev.shape[0]

  x0 = xprev[0]

  for iteration in range(num_iterations):

    x = cp.Variable((T, robot.stateDim))
    u = cp.Variable((T-1, robot.ctrlDim))

    objective = cp.Minimize(cp.sum_squares(u))

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
      A, B, y = jacobian(robot, xbar, ubar, data_neighbors_t)
      # simple version:
      constraints.append(
        x[t+1] == x[t] + dt * (y.numpy() + A.numpy() @ (x[t] - xbar.numpy()) + B.numpy() @ (u[t] - ubar.numpy()))
        )
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
      x_int[t+1] = x_int[t] + dt * (robot.f(x_int[t], uprev[t], data_neighbors_t))
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

    # objective = cp.Minimize(cp.sum(cp.abs(x[-1] - xf)))
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
    for t in range(0, T-1):
      xbar = xprev[t]
      ubar = uprev[t]

      # print(other_x)
      data_neighbors_t = get_data_neighbors(data_neighbors, t)
      # print(x_neighbors)
      A, B, y = jacobian(robot, xbar, ubar, data_neighbors_t)
      # simple version:
      constraints.append(
        x[t+1] == x[t] + dt * (y.numpy() + A.numpy() @ (x[t] - xbar.numpy()) + B.numpy() @ (u[t] - ubar.numpy()))
        )
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
      x_int[t+1] = x_int[t] + dt * (robot.f(x_int[t], uprev[t], data_neighbors_t))
    X_integration.append(x_int.detach())

    if prob.value < 1e-8:
      break

  return X, U, X_integration, prob.value