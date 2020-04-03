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

def jacobian(robot, xbar, ubar):
  xbar.requires_grad = True
  ubar.requires_grad = True
  y = robot.f(xbar, ubar)
  result = [torch.autograd.grad(outputs=[y], inputs=[xbar], grad_outputs=[unit], retain_graph=True, allow_unused=True)[0] for unit in unit_vectors(y.size(0))]
  jacobianA = torch.stack(result, dim=0)
  result = [torch.autograd.grad(outputs=[y], inputs=[ubar], grad_outputs=[unit], retain_graph=True, allow_unused=True)[0] for unit in unit_vectors(y.size(0))]
  jacobianB = torch.stack(result, dim=0)

  xbar.requires_grad = False
  ubar.requires_grad = False

  return jacobianA, jacobianB, y.detach()

def scp(robot, initial_x, initial_u, dt, trust_region=False, trust_x=2, trust_u=2, num_iterations=10):
  X, U = [], []
  X_integration = []

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
    ]

    # goal state constraint
    constraints.append( x[-1] == initial_x[-1] )

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
      A, B, y = jacobian(robot, xbar, ubar)
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

      if hasattr(robot, 'thrust_to_weight'):
        # collison check
        xbar = xprev[t]
        dist = np.linalg.norm([xbar[0]-xbar[4], xbar[1]-xbar[5]])
        constraints.extend([
        (x[t, 0]-xbar[4])*(xbar[0]-xbar[4]) + (x[t, 1]-xbar[5])*(xbar[1]-xbar[5]) >= robot.radius*dist,
        (x[t, 4]-xbar[0])*(xbar[4]-xbar[0]) + (x[t, 5]-xbar[1])*(xbar[5]-xbar[1]) >= robot.radius*dist
        ])

    for t in range(0, T-1):
      if hasattr(robot, 'u_min'):
        constraints.extend([
        robot.u_min <= u[t],
        u[t] <= robot.u_max
        ])
      if hasattr(robot, 'thrust_to_weight'):
        constraints.extend([
        u[t, 0]**2 + u[t, 1]**2 <= (robot.g*robot.thrust_to_weight)**2,
        u[t, 2]**2 + u[t, 3]**2 <= (robot.g*robot.thrust_to_weight)**2
        ])


    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(verbose=True, solver=cp.GUROBI, BarQCPConvTol=1e-8)

    xprev = torch.tensor(x.value, dtype=torch.float32)
    uprev = torch.tensor(u.value, dtype=torch.float32)
    X.append(xprev)
    U.append(uprev)

    x_int = torch.zeros((T, robot.stateDim))
    x_int[0] = x0
    for t in range(0, T-1):
        x_int[t+1] = x_int[t] + dt * (robot.f(x_int[t], uprev[t]))
    X_integration.append(x_int.detach())

  return X, U, X_integration

def scp_sequential(robot, initial_x, initial_u, dt, trust_region=False, trust_x=2, trust_u=2, num_iterations=10):
  X, U = [], []
  X_integration = []

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
    ]

    # goal state constraint
    constraints.append( x[-1] == initial_x[-1] )

    # sequential constraint: enforce x/u to be constant every other iteration
    for t in range(0, T):
      if iteration % 2 == 0:
        constraints.extend([
          x[t, 4:] == xprev[t, 4:]
        ])
      else:
        constraints.extend([
          x[t, :4] == xprev[t, :4]
        ])
    for t in range(0, T-1):
      if iteration % 2 == 0:
        constraints.extend([
          u[t, 2:] == uprev[t, 2:]
        ])
      else:
        constraints.extend([
          u[t, :2] == uprev[t, :2]
        ])

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
      A, B, y = jacobian(robot, xbar, ubar)
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

      if hasattr(robot, 'thrust_to_weight'):
        # collison check
        xbar = xprev[t]
        dist = np.linalg.norm([xbar[0]-xbar[4], xbar[1]-xbar[5]])
        constraints.extend([
        (x[t, 0]-xbar[4])*(xbar[0]-xbar[4]) + (x[t, 1]-xbar[5])*(xbar[1]-xbar[5]) >= robot.radius*dist,
        (x[t, 4]-xbar[0])*(xbar[4]-xbar[0]) + (x[t, 5]-xbar[1])*(xbar[5]-xbar[1]) >= robot.radius*dist
        ])

    for t in range(0, T-1):
      if hasattr(robot, 'u_min'):
        constraints.extend([
        robot.u_min <= u[t],
        u[t] <= robot.u_max
        ])
      if hasattr(robot, 'thrust_to_weight'):
        constraints.extend([
        u[t, 0]**2 + u[t, 1]**2 <= (robot.g*robot.thrust_to_weight)**2,
        u[t, 2]**2 + u[t, 3]**2 <= (robot.g*robot.thrust_to_weight)**2
        ])


    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(verbose=True, solver=cp.GUROBI, BarQCPConvTol=1e-8)

    xprev = torch.tensor(x.value, dtype=torch.float32)
    uprev = torch.tensor(u.value, dtype=torch.float32)
    X.append(xprev)
    U.append(uprev)

    x_int = torch.zeros((T, robot.stateDim))
    x_int[0] = x0
    for t in range(0, T-1):
        x_int[t+1] = x_int[t] + dt * (robot.f(x_int[t], uprev[t]))
    X_integration.append(x_int.detach())

  return X, U, X_integration

def scp_sequential_2(robot, initial_x, initial_u, dt, trust_region=False, trust_x=2, trust_u=2, num_iterations=10):
  '''
  This code only works for RobotTwoCrazyFlies2D now and needs a lot of improvement
  '''
  X, U = [], []
  X_integration = []

  xprev = initial_x
  uprev = initial_u
  T = xprev.shape[0]

  x0 = xprev[0]

  for iteration in range(num_iterations):
    if iteration % 2 == 0:
      cf = '1'
    else:
      cf = '2'

    x = cp.Variable((T, 4))
    u = cp.Variable((T-1, 2))

    objective = cp.Minimize(cp.sum_squares(u))

    if cf == '1':
      constraints = [ x[0] == x0[:4] ]
    else:
      constraints = [ x[0] == x0[4:] ]

    # goal state constraint
    if cf == '1':
      constraints.append( x[-1] == initial_x[-1, :4] )
    else:
      constraints.append( x[-1] == initial_x[-1, 4:] )      

    # trust region
    if trust_region:
      for t in range(0, T):
        if cf == '1':
          constraints.append(
            cp.abs(x[t] - xprev[t, :4]) <= trust_x
          )
        else:
          constraints.append(
            cp.abs(x[t] - xprev[t, 4:]) <= trust_x
          )
      for t in range(0, T-1):
        if cf == '1':
          constraints.append(
            cp.abs(u[t] - uprev[t, :2]) <= trust_u
          )
        else:
          constraints.append(
            cp.abs(u[t] - uprev[t, 2:]) <= trust_u
          )

    # dynamics constraints
    for t in range(0, T-1):
      xbar = xprev[t]
      ubar = uprev[t]
      A, B, y = jacobian(robot, xbar, ubar)
      # simple version:
      if cf == '1':
        constraints.append(
          x[t+1] == x[t] + dt * (y.numpy()[:4] + A.numpy()[:4,:4] @ (x[t] - xbar.numpy()[:4]) + B.numpy()[:4,:2] @ (u[t] - ubar.numpy()[:2]))
          )
      else:
        constraints.append(
          x[t+1] == x[t] + dt * (y.numpy()[4:] + A.numpy()[4:,4:] @ (x[t] - xbar.numpy()[4:]) + B.numpy()[4:,2:] @ (u[t] - ubar.numpy()[2:]))
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
      if cf == '1':
        constraints.extend([
          robot.x_min[:4] <= x[t],
          x[t] <= robot.x_max[:4]
          ])
        # collison check
        xbar = xprev[t]
        dist = np.linalg.norm([xbar[0]-xbar[4], xbar[1]-xbar[5]])
        constraints.extend([ (x[t, 0]-xbar[4])*(xbar[0]-xbar[4]) + (x[t, 1]-xbar[5])*(xbar[1]-xbar[5]) >= robot.radius*dist ])
      else:
        constraints.extend([
          robot.x_min[4:] <= x[t],
          x[t] <= robot.x_max[4:]
          ])
        # collison check
        xbar = xprev[t]
        dist = np.linalg.norm([xbar[0]-xbar[4], xbar[1]-xbar[5]])
        constraints.extend([ (x[t, 0]-xbar[0])*(xbar[4]-xbar[0]) + (x[t, 1]-xbar[1])*(xbar[5]-xbar[1]) >= robot.radius*dist ])
    
    for t in range(0, T-1):
      if cf == '1':
        constraints.extend([ u[t, 0]**2 + u[t, 1]**2 <= (robot.g*robot.thrust_to_weight)**2 ])
      else:
        constraints.extend([ u[t, 0]**2 + u[t, 1]**2 <= (robot.g*robot.thrust_to_weight)**2 ])

    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(verbose=True, solver=cp.GUROBI, BarQCPConvTol=1e-8)

    x_temp = torch.zeros((T, robot.stateDim))
    u_temp = torch.zeros((T-1, robot.ctrlDim))
    if cf == '1':
      x_temp[:, :4] = torch.tensor(x.value, dtype=torch.float32)
      u_temp[:, :2] = torch.tensor(u.value, dtype=torch.float32)
      x_temp[:, 4:] = xprev[:, 4:]
      u_temp[:, 2:] = uprev[:, 2:]
    else:
      x_temp[:, 4:] = torch.tensor(x.value, dtype=torch.float32)
      u_temp[:, 2:] = torch.tensor(u.value, dtype=torch.float32)
      x_temp[:, :4] = xprev[:, :4]
      u_temp[:, :2] = uprev[:, :2]

    xprev = x_temp
    uprev = u_temp
    X.append(xprev)
    U.append(uprev)

    x_int = torch.zeros((T, robot.stateDim))
    x_int[0] = x0
    for t in range(0, T-1):
        x_int[t+1] = x_int[t] + dt * (robot.f(x_int[t], uprev[t]))
    X_integration.append(x_int.detach())

  return X, U, X_integration