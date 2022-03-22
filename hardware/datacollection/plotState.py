import CF_functions as cff
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
import numpy.ma as ma
import math

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

# Convert quaternion to rotation matrix
def rotation_matrix(quat):
    rot_mat = np.ones([3,3])
    a = quat[0]**2
    b = quat[1]**2
    c = quat[2]**2
    d = quat[3]**2
    e = quat[0]*quat[1]
    f = quat[0]*quat[2]
    g = quat[0]*quat[3]
    h = quat[1]*quat[2]
    i = quat[1]*quat[3]
    j = quat[2]*quat[3]
    rot_mat[0,0] = a - b - c + d
    rot_mat[0,1] = 2 * (e - j)
    rot_mat[0,2] = 2 * (f + i)
    rot_mat[1,0] = 2 * (e + j)
    rot_mat[1,1] = -a + b - c + d
    rot_mat[1,2] = 2 * (h - g)
    rot_mat[2,0] = 2 * (f - i)
    rot_mat[2,1] = 2 * (h + g)
    rot_mat[2,2] = -a - b + c + d
   
    return rot_mat

def force2pwm(pwm, vbat):
  return C_00 + C_10*pwm + C_01*vbat + C_20*pwm**2 + C_11*vbat*pwm

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="logfile")
args = parser.parse_args()

# decode binary log data
logData = cff.decode(args.file)

time = logData['tick'] / 1e6 # us -> s

# compute rotations

quatState = np.array([quatdecompress(int(c)) for c in logData['stateEstimateZ.quat']])
# quatState = [[
#   logData['stateEstimate.qx'][i],
#   logData['stateEstimate.qy'][i],
#   logData['stateEstimate.qz'][i],
#   logData['stateEstimate.qw'][i]] for i in range(len(time))]
state_rpy = np.array([quat2rpy(q) for q in quatState])

# compute roll/pitch/yaw components of thrust mixing

if 'motor.torquez' in logData:
  thrust_to_torque = 0.006
  arm_length = 0.046 # m
  yawpart = -0.25 * logData['motor.torquez'] / thrust_to_torque
  arm = 0.707106781 * arm_length
  rollpart = 0.25 / arm * logData['motor.torquex'];
  pitchpart = 0.25 / arm * logData['motor.torquey'];


# set window background to white
# plt.rcParams['figure.facecolor'] = 'w'
    
# number of columns and rows for suplot
plotCols = 3
plotRows = 4

# current plot for simple subplot usage
plotCurrent = 0

for i, axis in enumerate(['x', 'y', 'z']):

  plotCurrent = i + 1
  plt.subplot(plotRows, plotCols, plotCurrent)
  plt.plot(time, logData['stateEstimateZ.' + axis] / 1000.0, '-', label='state')
  plt.plot(time, logData['ctrltargetZ.' + axis] / 1000.0, '-', label='target')
  plt.xlabel('Time [s]')
  plt.ylabel('Position {} [m]'.format(axis))
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  plotCurrent = i + 4
  plt.subplot(plotRows, plotCols, plotCurrent)
  plt.plot(time, logData['stateEstimateZ.v' + axis] / 1000.0, '-', label='state')
  plt.plot(time, logData['ctrltargetZ.v' + axis] / 1000.0, '-', label='target')
  plt.xlabel('Time [s]')
  plt.ylabel('Velocity {} [m/s]'.format(axis))
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  plotCurrent = i + 7
  plt.subplot(plotRows, plotCols, plotCurrent)
  plt.plot(time, np.degrees(state_rpy[:,i]), '-', label='state')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle {} [deg]'.format(axis))
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

plotCurrent = 10
plt.subplot(plotRows, plotCols, plotCurrent)
plt.plot(time, logData['motor.f1'] / 9.81 * 1000, '-', label='f1')
plt.plot(time, logData['motor.f2'] / 9.81 * 1000, '-', label='f2')
plt.plot(time, logData['motor.f3'] / 9.81 * 1000, '-', label='f3')
plt.plot(time, logData['motor.f4'] / 9.81 * 1000, '-', label='f4')
plt.plot(time, logData['pwm.maxThrust'], '--')

plt.xlabel('Time [s]')
plt.ylabel('Desired motor force [g]'.format(axis))
plt.legend(loc=9, ncol=3, borderaxespad=0.)

plotCurrent = 11
plt.subplot(plotRows, plotCols, plotCurrent)
# estimate Fa based on collected data:
mass = 0.032
C_00 = 11.093358483549203
C_10 = -39.08104165843915
C_01 = -9.525647087583181
C_20 = 20.573302305476638
C_11 = 38.42885066644033
if "cf100" in args.file or "cf101" in args.file or "cf102" in args.file:
  mass = 0.067
  C_00 = 44.10386631845999
  C_10 = -122.51151800146272
  C_01 = -36.18484254283743
  C_20 = 53.10772568607133
  C_11 = 107.6819263349139
if "cf200" in args.file:
  mass = 0.032
  C_00 = 14.639083451431064
  C_10 = -49.58925346670507
  C_01 = -15.11436310327852
  C_20 = 25.806716788604707
  C_11 = 54.00127729445893

acc = np.column_stack((
  logData['stateEstimateZ.ax'] / 1000.0,
  logData['stateEstimateZ.ay'] / 1000.0,
  logData['stateEstimateZ.az'] / 1000.0 - 9.81))

vel = np.column_stack((
  logData['stateEstimateZ.vx'] / 1000.0,
  logData['stateEstimateZ.vy'] / 1000.0,
  logData['stateEstimateZ.vz'] / 1000.0))

print(vel.shape, time.shape)
acc2 = np.diff(vel, axis=0) / np.column_stack((np.diff(time), np.diff(time), np.diff(time)))

# thrust_fw = logData['motor.f1'] + logData['motor.f2'] + logData['motor.f3'] + logData['motor.f4']
# Estimate thrust using PWM model (to account for motor saturation; output in grams)
force_pwm_1 = force2pwm(logData['pwm.m1_pwm'] / 65536, logData['pm.vbatMV'] / 1000 / 4.2)
force_pwm_2 = force2pwm(logData['pwm.m2_pwm'] / 65536, logData['pm.vbatMV'] / 1000 / 4.2)
force_pwm_3 = force2pwm(logData['pwm.m3_pwm'] / 65536, logData['pm.vbatMV'] / 1000 / 4.2)
force_pwm_4 = force2pwm(logData['pwm.m4_pwm'] / 65536, logData['pm.vbatMV'] / 1000 / 4.2)
thrust = (force_pwm_1 + force_pwm_2 + force_pwm_3 + force_pwm_4) / 1000 * 9.81

# consider delay
thrust_delay = np.zeros(len(thrust))
thrust_delay[0] = thrust[0]
for i in range(len(thrust)-1):
  thrust_delay[i+1] = (1-0.16)*thrust_delay[i] + 0.16*thrust[i] 

fa = []
for q, a, T in zip(quatState, acc, thrust):
  fu = np.array([0, 0, T])
  fa.append(mass * a - mass * np.array([0,0,-9.81]) - rotation_matrix(q) @ fu)
fa = np.array(fa)

fa_delay = []
for q, a, T in zip(quatState, acc, thrust_delay):
  fu = np.array([0, 0, T])
  fa_delay.append(mass * a - mass * np.array([0,0,-9.81]) - rotation_matrix(q) @ fu)
fa_delay = np.array(fa_delay)

# plt.plot(time, fa[:,2] / 9.81 * 1000, '-', label='Fa.z')
plt.plot(time, fa_delay[:,2] / 9.81 * 1000, '-', label='Fa.z (delay)')
# plt.plot(time, acc[:,2])
# plt.plot(time[1:], acc2[:,2])
if "ctrlFa.Faz" in logData:
  plt.plot(time, logData['ctrlFa.Faz'], '-', label='Fa.z (NN)')

plt.xlabel('Time [s]')
plt.ylabel('Fa')
plt.legend(loc=9, ncol=3, borderaxespad=0.)


# plotCurrent = 12
# plt.subplot(plotRows, plotCols, plotCurrent)
# plt.plot(time, logData['motor.torquex'], '-', label='x')
# plt.plot(time, logData['motor.torquey'], '-', label='y')
# plt.plot(time, logData['motor.torquez'], '-', label='z')
# plt.xlabel('Time [s]')
# plt.ylabel('Torque [Nm]'.format(axis))
# plt.legend(loc=9, ncol=3, borderaxespad=0.)

plotCurrent = 12
plt.subplot(plotRows, plotCols, plotCurrent)
plt.plot(time, logData['motor.thrust'], '-')
plt.xlabel('Time [s]')
plt.ylabel('Thrust [N]')

# if 'motor.torquez' in logData:
#   plotCurrent = 12
#   plt.subplot(plotRows, plotCols, plotCurrent)
#   # plt.plot(time, rollpart / 9.81 * 1000, '-', label='x')
#   # plt.plot(time, pitchpart / 9.81 * 1000, '-', label='y')
#   # plt.plot(time, yawpart / 9.81 * 1000, '-', label='z')
#   plt.stackplot(time, np.abs(rollpart / 9.81 * 1000), np.abs(pitchpart / 9.81 * 1000), np.abs(yawpart / 9.81 * 1000), labels=["roll","pitch", "yaw"])

#   plt.xlabel('Time [s]')
#   plt.ylabel('Thrust mixing [g]'.format(axis))
#   plt.legend(loc=9, ncol=3, borderaxespad=0.)

plt.show()
