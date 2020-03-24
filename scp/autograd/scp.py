import cvxpy as cp
# import numpy as np
import autograd.numpy as np  # Thinly-wrapped numpy
import scipy
from autograd import grad, elementwise_grad, jacobian

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_pdf


def scp(robot, initialFile = None, initialU = None, initialX = None, dt = None, goalState = None, goalPos = None, pdfFile = None):
  partialFx = jacobian(robot.f, 0)
  partialFu = jacobian(robot.f, 1)

  def constructA(xbar, ubar):
    return partialFx(xbar, ubar)

  def constructB(xbar, ubar):
    return partialFu(xbar, ubar)

  if initialFile is not None:
    data = np.loadtxt(
      initialFile,
      skiprows=1,
      delimiter=',')
    xprev = data[:,robot.idxState:robot.idxState+robot.stateDim]
    uprev = data[:,robot.idxCtrl:robot.idxCtrl+robot.ctrlDim]
    T = data.shape[0]
    dt = data[0,-1]
    print(T)
    print(xprev, uprev)
  else:
    xprev = initialX
    uprev = initialU
    T = xprev.shape[0]

  x0 = xprev[0]

  if pdfFile is not None:
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdfFile)

  objectiveValues = []
  xChanges = []
  uChanges = []
  try:
    obj = 'minimizeError'

    for iteration in range(0, 10):

      x = cp.Variable((T, robot.stateDim))
      u = cp.Variable((T, robot.ctrlDim))

      if obj == 'minimizeError':
        delta = cp.Variable()
        objective = cp.Minimize(delta)
      else:
        # objective = cp.Minimize(cp.sum_squares(u) + 10 * cp.sum_squares(x[:,3:5]))
        objective = cp.Minimize(cp.sum_squares(u))
        # objective = cp.Minimize(cp.sum_squares(u) + 10 * cp.sum(x[:,4]))
      constraints = [
        x[0] == x0, # initial state constraint
      ]

      if obj == 'minimizeError':
        if goalState is not None:
          constraints.append( cp.abs(x[-1] - goalState) <= delta )
        else:
          constraints.append(cp.abs(x[-1,0:2] - goalPos) <= delta )
      else:
        if goalState is not None:
          constraints.append( x[-1] == goalState )
        else:
          constraints.append( x[-1,0:2] == goalPos )

      # trust region
      for t in range(0, T):
        constraints.append(
          cp.abs(x[t] - xprev[t]) <= 2 #0.1
        )
        constraints.append(
          cp.abs(u[t] - uprev[t]) <= 2 #0.1
        )

      # dynamics constraints
      for t in range(0, T-1):
        xbar = xprev[t]
        ubar = uprev[t]
        A = constructA(xbar, ubar)
        B = constructB(xbar, ubar)
        # print(xbar, ubar, A, B)
        # print(f(xbar, ubar))
        # # simple version:
        constraints.append(
          x[t+1] == x[t] + dt * (robot.f(xbar, ubar) + A @ (x[t] - xbar) + B @ (u[t] - ubar))
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
      try:
        result = prob.solve(verbose=True,solver=cp.GUROBI, BarQCPConvTol=1e-8)
      except cvxpy.error.SolverError:
        return

      if result is None:
        return

      objectiveValues.append(result)
      xChanges.append(np.linalg.norm(x.value - xprev))
      uChanges.append(np.linalg.norm(u.value - uprev))

      if result < 1e-6:
        obj = 'minimizeU'

      # The optimal value for x is stored in `x.value`.
      # print(x.value)
      # print(u.value)

      # compute forward propagated value
      xprop = np.empty(x.value.shape)
      xprop[0] = x0
      for t in range(0, T-1):
        xprop[t+1] = xprop[t] + dt * robot.f(xprop[t], u.value[t])

      # print(xprop)
      if pdfFile is not None:
        if robot.is2D:
          fig, ax = plt.subplots()
          ax.plot(xprev[:,0], xprev[:,1], label="input")
          ax.plot(x.value[:,0], x.value[:,1], label="opt")
          ax.plot(xprop[:,0], xprop[:,1], label="opt forward prop")
        else:
          fig = plt.figure()
          ax = fig.add_subplot(111, projection='3d')
          ax.plot(xprev[:,0], xprev[:,1], xprev[:,2], label="input")
          ax.plot(x.value[:,0], x.value[:,1], x.value[:,2], label="opt")
          # ax.plot(xprop[:,0], xprop[:,1], xprop[:,2], label="opt forward prop")

        plt.legend()
        # plt.show()
        pdf.savefig(fig)
        plt.close(fig)

      xprev = np.array(x.value)
      uprev = np.array(u.value)
  except:
    raise
  finally:
    # print(xprev)
    # print(uprev)
    if pdfFile is not None:
      fig, ax = plt.subplots()
      ax.plot(np.arange(1,len(objectiveValues)+1), objectiveValues, '*-', label='cost')
      ax.plot(np.arange(1,len(objectiveValues)+1), xChanges, '*-', label='|x-xp|')
      ax.plot(np.arange(1,len(objectiveValues)+1), uChanges, '*-', label='|u-up|')
      plt.legend()
      pdf.savefig(fig)
      pdf.close()

  if obj == 'minimizeU':
    return xprev, uprev