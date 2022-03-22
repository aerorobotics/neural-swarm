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

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="logfile")
parser.add_argument("--axis", type=str, default='x', help="logfile")
args = parser.parse_args()

# decode binary log data
logData = cff.decode(args.file)

# set window background to white
plt.rcParams['figure.facecolor'] = 'w'
    
# number of columns and rows for suplot
plotCols = 3
plotRows = 6

# current plot for simple subplot usage
plotCurrent = 0

time = logData['tick'] / 1e6

for i, axis in enumerate(['x', 'y', 'z']):

	plotCurrent = i + 1
	plt.subplot(plotRows, plotCols, plotCurrent)
	plt.plot(time, np.degrees(logData['ctrlSJC.q' + axis]), '-', label='state')
	plt.plot(time, np.degrees(logData['ctrlSJC.qr' + axis]), '-', label='target')
	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Angle [deg]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

	plotCurrent = i + 4
	plt.subplot(plotRows, plotCols, plotCurrent)
	plt.plot(time, np.degrees(logData['ctrlSJC.omega' + axis]), '-', label='w')
	plt.plot(time, np.degrees(logData['ctrlSJC.omegar' + axis]), '-', label='w_r')
	# plt.plot(time, np.degrees(logData['ctrlSJC.omega' + axis] - logData['ctrlSJC.omegar' + axis]), '-', label='w_e')
	plt.plot(time, np.degrees(logData['ctrlSJC.i_error_att' + axis]), '-', label='accumulated error')
	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Angular velocity [deg/s]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

	plotCurrent = i + 7
	plt.subplot(plotRows, plotCols, plotCurrent)
	plt.plot(time, np.degrees(logData['ctrlSJC.domega' + axis]), '-', label='w dot')

	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Angular acceleration [deg/s^2]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

	plotCurrent = i + 10
	plt.subplot(plotRows, plotCols, plotCurrent)
	# plt.plot(time, logData['stateEstimateZ.' + axis] / 1000.0, '-', label='state')
	# plt.plot(time, logData['ctrltargetZ.' + axis] / 1000.0, '-', label='target')
	plt.plot(time, logData['stateEstimateZ.' + axis] / 1000.0 - logData['ctrltargetZ.' + axis] / 1000.0, '-', label='error')
	plt.plot(time, logData['ctrlSJC.i_error_pos' + axis], '-', label='accumulated error')
	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Position [m]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

	plotCurrent = i + 13
	plt.subplot(plotRows, plotCols, plotCurrent)
	plt.plot(time, logData['stateEstimateZ.v' + axis] / 1000.0, '-', label='state')
	plt.plot(time, logData['ctrltargetZ.v' + axis] / 1000.0, '-', label='target')
	plt.xlabel('RTOS Ticks [ms]')
	plt.ylabel('Velocity [m/s]')
	plt.legend(loc=9, ncol=3, borderaxespad=0.)

plotCurrent = 16
plt.subplot(plotRows, plotCols, plotCurrent)
plt.plot(time, logData['motor.f1'] / 9.81 * 1000, '-', label='f1')
plt.plot(time, logData['motor.f2'] / 9.81 * 1000, '-', label='f2')
plt.plot(time, logData['motor.f3'] / 9.81 * 1000, '-', label='f3')
plt.plot(time, logData['motor.f4'] / 9.81 * 1000, '-', label='f4')
plt.plot(time, logData['pwm.maxThrust'], '--')

plt.xlabel('Time [s]')
plt.ylabel('Desired motor force [g]'.format(axis))
plt.legend(loc=9, ncol=3, borderaxespad=0.)

plotCurrent = 17
plt.subplot(plotRows, plotCols, plotCurrent)

# estimate Fa based on collected data:
mass = 0.033
# mass = 0.067

quatState = np.array([quatdecompress(int(c)) for c in logData['stateEstimateZ.quat']])
acc = np.column_stack((
  logData['stateEstimateZ.ax'] / 1000.0,
  logData['stateEstimateZ.ay'] / 1000.0,
  logData['stateEstimateZ.az'] / 1000.0 - 9.81))


Thrust = logData['motor.f1'] + logData['motor.f2'] + logData['motor.f3'] + logData['motor.f4']

fa = []
for q, a, T in zip(quatState, acc, Thrust):
  fu = np.array([0, 0, T])
  fa.append(mass * a - mass * np.array([0,0,-9.81]) - rotation_matrix(q) @ fu)

fa = np.array(fa)

plt.plot(time, fa[:,2] / 9.81 * 1000, '-', label='est. Fa')

plotCurrent = 18
plt.subplot(plotRows, plotCols, plotCurrent)
plt.plot(time, acc[:,2], '-', label='acc (IMU)')
# plt.plot(time[1:], acc2[:,2], '-', label='acc (IMU)')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]'.format(axis))


	# plotCurrent = i + 10
	# plt.subplot(plotRows, plotCols, plotCurrent)
	# plt.plot(time, logData['pwm.m1_pwm'] / 65536.0, '-', label='m1')
	# plt.plot(time, logData['pwm.m2_pwm'] / 65536.0, '-', label='m2')
	# plt.plot(time, logData['pwm.m3_pwm'] / 65536.0, '-', label='m3')
	# plt.plot(time, logData['pwm.m4_pwm'] / 65536.0, '-', label='m4')
	# # plt.plot(time, logData['motor.saturation'], '-', label='saturation')
	# # plt.plot(time, logData['ctrlSJC.torquex'], '-', label='torquex')
	# # plt.plot(time, logData['ctrlSJC.torquey'], '-', label='torquey')
	# # plt.plot(time, logData['ctrlSJC.torquez'], '-', label='torquez')
	# plt.xlabel('RTOS Ticks [ms]')
	# plt.ylabel('Normalized motor output')
	# plt.legend(loc=9, ncol=3, borderaxespad=0.)

plt.show()
