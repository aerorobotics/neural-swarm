import torch
import cvxpy as cp

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

def scp(robot, initial_x, initial_u, dt, trust_x=2, trust_u=2, num_iterations=10):

  xprev = initial_x
  uprev = initial_u
  T = xprev.shape[0]

  x0 = xprev[0]

  for iteration in range(num_iterations):

    x = cp.Variable((T, robot.stateDim))
    u = cp.Variable((T, robot.ctrlDim))

    objective = cp.Minimize(cp.sum_squares(u))

    constraints = [
      x[0] == x0, # initial state constraint
    ]

    # goal state constraint
    constraints.append( x[-1] == initial_x[-1] )

    # trust region
    for t in range(0, T):
      constraints.append(
        cp.abs(x[t] - xprev[t]) <= trust_x
      )
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
        x[t] <= robot.x_max,
        robot.u_min <= u[t],
        u[t] <= robot.u_max
        ])

    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve(verbose=True,solver=cp.GUROBI, BarQCPConvTol=1e-8)

    xprev = torch.tensor(x.value, dtype=torch.float32)
    uprev = torch.tensor(u.value, dtype=torch.float32)

  return xprev, uprev
