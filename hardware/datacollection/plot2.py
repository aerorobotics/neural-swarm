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

fig, ax = plt.subplots(3, 1)

for suffix in ["baseline", "nn"]:

  # decode binary log data
  logData = cff.decode("{}_{}".format(args.file, suffix))

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
  error_z = np.abs(pos[:,2] - pos_d[:,2])

  print("Error {}: mean {:.3f} std {:.3f} max {:.3f} max-z {:.3f}".format(suffix, np.mean(error), np.std(error), np.max(error), np.max(error_z)))


  for i in [0,1,2]:
    ax[i].plot(time, pos[:,i], label="{}".format(suffix))
    ax[i].plot(time, pos_d[:,i], linestyle='--', label="{} des".format(suffix))

ax[0].legend()
plt.show()

#   for k, robot in enumerate(robots):
#     X = getattr(robot, 'X_' + field)
#     U = getattr(robot, 'U_' + field)
#     if use3D:
#       line = ax[0].plot(torch.norm(X[:,3:6], dim=1), label="cf{}".format(k))
#     else:
#       line = ax[0].plot(torch.norm(X[:,2:4], dim=1), label="cf{}".format(k))
#     ax[1].plot(torch.norm(U, dim=1), line[0].get_color())
#     colors.append(line[0].get_color())
#   ax[0].legend()
#   ax[0].set_title('{} - Velocity'.format(name))
#   ax[1].legend()
#   ax[1].set_title('{} - Acceleration'.format(name))
#   pp.savefig(fig)
#   plt.close(fig)




# for i in [0,1,2]:
#   error = np.abs(pos[:,i] - pos_d[:,i])

#   print("Error: mean {:.3f} std {:.3f} max {:.3f}".format(np.mean(error), np.std(error), np.max(error)))

#   plt.plot(time, error)

#   plt.show()
