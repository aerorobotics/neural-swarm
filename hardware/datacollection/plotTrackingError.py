import CF_functions as cff
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np
import numpy.ma as ma
import math

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="logfile")
args = parser.parse_args()

# decode binary log data
logData = cff.decode(args.file)

time = logData['tick'] / 1e6 # us -> s

pos = np.column_stack((
  logData['stateEstimateZ.x'] / 1000.0,
  logData['stateEstimateZ.y'] / 1000.0,
  logData['stateEstimateZ.z'] / 1000.0))

pos_d = np.column_stack((
  logData['ctrltargetZ.x'] / 1000.0,
  logData['ctrltargetZ.y'] / 1000.0,
  logData['ctrltargetZ.z'] / 1000.0))

error = np.linalg.norm(pos - pos_d, axis=1)

for i in [0,1,2]:
  error = np.abs(pos[:,i] - pos_d[:,i])

  error1 = error[0:int(len(error) / 2)]
  print("Error (baseline): mean {:.3f} std {:.3f} max {:.3f}".format(np.mean(error1), np.std(error1), np.max(error1)))

  error2 = error[int(len(error) / 2):-1]
  print("Error (NN): mean {:.3f} std {:.3f} max {:.3f}".format(np.mean(error2), np.std(error2), np.max(error2)))

  plt.plot(time, error)

  plt.show()
