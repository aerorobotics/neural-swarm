import CF_functions as cff
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
import numpy.ma as ma
import math
import glob

# decompress a quaternion, see quatcompress.h in firmware
def quatdecompress(comp):
  q = np.zeros(4)
  mask = (1 << 9) - 1
  i_largest = comp >> 30
  sum_squares = 0
  for i in range(3, -1, -1):
    if i != i_largest:
      mag = comp & mask
      negbit = (comp >> 9) & 0x1
      comp = comp >> 10
      q[i] = mag / mask / math.sqrt(2)
      if negbit == 1:
        q[i] = -q[i]
      sum_squares += q[i] * q[i]
  q[i_largest] = math.sqrt(1.0 - sum_squares)
  return q

# convert quaternion to (roll, pitch, yaw) Euler angles using Tait-Bryan convention
# (yaw, then pitch about new pitch axis, then roll about new roll axis)
# assume q = [x,y,z,w] order
def quat2rpy(q):
  rpy = np.zeros(3)
  rpy[0] = math.atan2(2 * (q[3] * q[0] + q[1] * q[2]), 1 - 2 * (q[0] * q[0] + q[1] * q[1]))
  rpy[1] = math.asin(2 * (q[3] * q[1] - q[0] * q[2]))
  rpy[2] = math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
  return rpy

# see Foerster, Hamer, D'Andrea, equation 3.3
# pwm in range 0...65536
def pwm2thrust(pwm):
  return 2.130295e-11 * pwm**2 + 1.032633e-6 * pwm + 5.484560e-4


parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str, help="folder")
args = parser.parse_args()

for file in glob.glob(args.folder + "/cf*"):
  print(file)

  # decode binary log data
  logData = cff.decode(file)

  quatZ = np.array([quatdecompress(int(c)) for c in logData['stateEstimateZ.quat']])
  # quat = np.column_stack((logData['stateEstimate.qx'], logData['stateEstimate.qy'], logData['stateEstimate.qz'], logData['stateEstimate.qw']))
  # print(quat)

  rpyZ = np.array([quat2rpy(q) for q in quatZ])
  # rpy = np.array([quat2rpy(q) for q in quat])

  thrust1 = np.array([pwm2thrust(pwm) for pwm in logData['pwm.m1_pwm']])
  thrust2 = np.array([pwm2thrust(pwm) for pwm in logData['pwm.m2_pwm']])
  thrust3 = np.array([pwm2thrust(pwm) for pwm in logData['pwm.m3_pwm']])
  thrust4 = np.array([pwm2thrust(pwm) for pwm in logData['pwm.m4_pwm']])


  # convert to matrix
  data = np.column_stack((
    logData['tick'], # ms
    logData['stateEstimateZ.x'] / 1000, # convert mm->m
    logData['stateEstimateZ.y'] / 1000, # convert mm->m
    logData['stateEstimateZ.z'] / 1000, # convert mm->m
    logData['stateEstimateZ.vx'] / 1000,# convert mm/s -> m/s
    logData['stateEstimateZ.vy'] / 1000,# convert mm/s -> m/s
    logData['stateEstimateZ.vz'] / 1000,# convert mm/s -> m/s
    logData['stateEstimateZ.ax'] / 1000,# convert mm/s^2 -> m/s^2
    logData['stateEstimateZ.ay'] / 1000,# convert mm/s^2 -> m/s^2
    logData['stateEstimateZ.az'] / 1000,# convert mm/s^2 -> m/s^2
    quatZ[:,0], # qx
    quatZ[:,1], # qy
    quatZ[:,2], # qz
    quatZ[:,3], # qw
    logData['motor.f1'], # convert internal value to N
    logData['motor.f2'], # convert internal value to N
    logData['motor.f3'], # convert internal value to N
    logData['motor.f4'],  # convert internal value to N
    logData['pwm.m1_pwm'] / 65536,
    logData['pwm.m2_pwm'] / 65536,
    logData['pwm.m3_pwm'] / 65536,
    logData['pwm.m4_pwm'] / 65536,
    logData['pm.vbatMV'] / 1000,
    logData['motor.thrust'] / 9.81 * 1000.0,
    logData['motor.torquex'],
    logData['motor.torquey'],
    logData['motor.torquez'],
    logData['stateEstimateZ.rateRoll'] / 1000.0,
    logData['stateEstimateZ.ratePitch'] / 1000.0,
    logData['stateEstimateZ.rateYaw'] / 1000.0,
    ))

  tau_u = np.column_stack((
    logData['motor.torquex'] / 9.81 * 1000.0,
    logData['motor.torquey'] / 9.81 * 1000.0,
    logData['motor.torquez'] / 9.81 * 1000.0))

  firstCrash = np.argmax(np.abs(rpyZ[:,0]) > np.radians(45))
  # print(file)
  # print(firstCrash)
  # print(rpyZ[:,0])
  # exit()

  # data = data[0:firstCrash]

  np.savetxt(file + ".csv", data, delimiter=',', header="time[ms],x[m],y[m],z[m],vx[m/s],vy[m/s],vz[m/s],ax[m/s^2],ay[m/s^2],az[m/s^2],qx,qy,qz,qw,u1[N],u2[N],u3[N],u4[N],u1[PWM],u2[PWM],u3[PWM],u4[PWM],Vbat[V],thrust_des[g],torquex[Nm],torquey[Nm],torquez[Nm],wx[rad/s],wy[rad/s],wz[rad/s]")


