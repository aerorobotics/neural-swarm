from robots import RobotDubinsCar, RobotAirplane
import autograd.numpy as np  # Thinly-wrapped numpy

if __name__ == '__main__':
  # robot = RobotDubinsCar(1, 1)

  # x = np.array([1,2,3], dtype=np.float32)
  # u = np.array([-0.5, 0.7], dtype=np.float32)

  # print(robot.f(x, u))

  robot = RobotAirplane()
  x = np.array(robot.x_min, dtype=np.float32)
  u = np.array(robot.u_min, dtype=np.float32)

  print(robot.f(x,u))

  # test jacobian computation
  from autograd import grad, elementwise_grad, jacobian

  partialFx = jacobian(robot.f, 0)
  partialFu = jacobian(robot.f, 1)

  def constructA(xbar, ubar):
    return partialFx(xbar, ubar)

  def constructB(xbar, ubar):
    return partialFu(xbar, ubar)

  A = constructA(x, u)
  B = constructB(x, u)

  print('A', A)
  print('B', B)

