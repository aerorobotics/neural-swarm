import CF_functions as cff
# import re
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import rowan
# import numpy.ma as ma
import math
import yaml
import os

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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("folder", type=str, help="folder with validation results")
  args = parser.parse_args()

  with open(os.path.join(args.folder, 'validation_config.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

  pp = PdfPages("result.pdf")

  for k, config in enumerate(configs):
    print(config["name"])

    # We are only interested in the lowest CF for now
    cfid = config["ids"][0]
    filename = "cf{0}_{1:02}".format(cfid, k)
    print(filename)

    # decode binary log data
    logData = cff.decode(filename)

    # find time when NN is switched on
    if logData['ctrlFa.Faz'][0] == 0:
      startIdx = 0
    else:
      startIdx = np.where(logData['ctrlFa.Faz'] == 0.)[0][0]

    nnIdx = np.where(logData['ctrlFa.Faz'][startIdx:] != 0.)[0][0]

    time = logData['tick'] / 1e6 # us -> s

    nnTime = time[nnIdx:] - time[nnIdx]

    quatState = np.array([quatdecompress(int(c)) for c in logData['stateEstimateZ.quat']])


    # compute Fa (considering motor delay)
    mass = 0.033
    if cfid >= 100 and cfid < 200:
      mass = 0.067

    acc = np.column_stack((
      logData['stateEstimateZ.ax'] / 1000.0,
      logData['stateEstimateZ.ay'] / 1000.0,
      logData['stateEstimateZ.az'] / 1000.0 - 9.81))

    vel = np.column_stack((
      logData['stateEstimateZ.vx'] / 1000.0,
      logData['stateEstimateZ.vy'] / 1000.0,
      logData['stateEstimateZ.vz'] / 1000.0))

    thrust = logData['motor.f1'] + logData['motor.f2'] + logData['motor.f3'] + logData['motor.f4']

    thrust_delay = np.zeros(len(thrust))
    thrust_delay[0] = thrust[0]
    for i in range(len(thrust)-1):
      thrust_delay[i+1] = (1-0.16)*thrust_delay[i] + 0.16*thrust[i] 

    fa_delay = []
    for q, a, T in zip(quatState, acc, thrust_delay):
      fu = np.array([0, 0, T])
      fa_delay.append(mass * a - mass * np.array([0,0,-9.81]) - rowan.rotate(q, fu))
    fa_delay = np.array(fa_delay)

    # compute stats
    config['height_error_baseline'] = logData['stateEstimateZ.z'][0:nnIdx] / 1000.0 - logData['ctrltargetZ.z'][0:nnIdx] / 1000.0
    config['height_error_nn'] = logData['stateEstimateZ.z'][nnIdx:] / 1000.0 - logData['ctrltargetZ.z'][nnIdx:] / 1000.0
    config['fa_error'] = logData['ctrlFa.Faz'][nnIdx:] - fa_delay[nnIdx:,2] / 9.81 * 1000

    # Plot height
    fig, ax = plt.subplots(3,1, squeeze=False)

    fig.suptitle(config["name"], fontsize=16)

    ax[0,0].plot(time[0:nnIdx], logData['stateEstimateZ.z'][0:nnIdx] / 1000.0, '-', label='baseline')
    ax[0,0].plot(nnTime, logData['stateEstimateZ.z'][nnIdx:] / 1000.0, '-', label='NN')
    ax[0,0].plot(time[0:nnIdx], logData['ctrltargetZ.z'][0:nnIdx] / 1000.0, '-', label='target')
    ax[0,0].set_xlabel('Time [s]')
    ax[0,0].set_ylabel('Height [m]')
    ax[0,0].legend()

    # Plot Fa
    ax[1,0].plot(nnTime, fa_delay[nnIdx:,2] / 9.81 * 1000, '-', label='Fa.z (delay)')
    ax[1,0].plot(nnTime, logData['ctrlFa.Faz'][nnIdx:], '-', label='Fa.z (NN)')
    ax[1,0].set_xlabel('Time [s]')
    ax[1,0].set_ylabel('Fa [g]')
    ax[1,0].legend()

    # Plot motor output and saturation
    ax[2,0].plot(time, logData['motor.f1'] / 9.81 * 1000, '-', label='f1')
    ax[2,0].plot(time, logData['motor.f2'] / 9.81 * 1000, '-', label='f2')
    ax[2,0].plot(time, logData['motor.f3'] / 9.81 * 1000, '-', label='f3')
    ax[2,0].plot(time, logData['motor.f4'] / 9.81 * 1000, '-', label='f4')
    ax[2,0].plot(time, logData['pwm.maxThrust'], '--')
    ax[2,0].set_xlabel('Time [s]')
    ax[2,0].set_ylabel('Desired motor force [g]')
    ax[2,0].legend()

    pp.savefig(fig)
    plt.close(fig)

  fig, ax = plt.subplots()
  ax.set_title("Fa error [g]")
  ax.boxplot([config['fa_error'] for config in configs])
  ax.set_xticklabels([config['name'] for config in configs])
  pp.savefig(fig)
  plt.close(fig)

  fig, ax = plt.subplots()
  ax.set_title("Height error [m]")
  data = []
  labels = []
  for k, config in enumerate(configs):
    data.append(config['height_error_baseline'])
    data.append(config['height_error_nn'])
    labels.append(config['name'])

  ax.boxplot(data)
  ylim = ax.get_ylim()

  xticks = []
  vlines = []
  for k, config in enumerate(configs):
    ax.text(0.5+2*k, ylim[0], "{0:.1f} → {1:.1f} cm".format(
      np.amax(np.absolute(config['height_error_baseline'])) * 100,
      np.amax(np.absolute(config['height_error_nn'])) * 100))
    xticks.append(2*k+1.5)
    vlines.append(2*k+2.5)

  ax.set_xticks(xticks)
  ax.set_xticklabels(labels)
  ax.vlines(vlines,ymin=ylim[0],ymax=ylim[1])
  pp.savefig(fig)
  plt.close(fig)



  pp.close()
  subprocess.call(["xdg-open", "result.pdf"])