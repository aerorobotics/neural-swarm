#!/usr/bin/env python

import numpy as np

from pycrazyswarm import *
import uav_trajectory

# # SS
# Ids = [50, 51]
# Heights = [0.6, 0.9]
# Radius = 0.3

# SL
Ids = [50, 101]
Heights = [0.6, 1.1]
Radius = 0.3

# # 3 agent case
# Ids = [50, 101, 102]
# # Heights = [0.6, 0.85, 1.1]
# # Heights = [0.5, 0.8, 1.1]
# # Heights = [0.5, 0.9, 1.1]
# Heights = [0.8, 0.5, 1.1]
# Radius = 0.35

# # 4 agent case
# Ids = [200, 201, 204, 203]
# Heights = [0.35, 0.6, 0.85, 1.1]
# Radius = 0.4

# # 5 agent case
# Ids = [204, 203, 202, 201, 200]
# Heights = [0.3, 0.5, 0.7, 0.9, 1.1]
# Radius = 0.4

# SwapTimes = [4,3,2]
# SwapTimes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# SwapTimes = [3, 3, 3]
SwapTimes = [3,2]

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # allcfs.setParam("stabilizer/controller", 5) # use Lee controller

    allcfs.takeoff(targetHeight=np.min(Heights), duration=3.0)
    timeHelper.sleep(3.5)

    # go to initial positions
    angles = np.linspace(0, 2*np.pi, 2 * len(Ids), endpoint=False)
    # print(angles, angles[0:len(Ids)], angles[len(Ids):])
    for angle, cfid, height in zip(angles[0:len(Ids)], Ids, Heights):
        pos = np.array([np.sin(angle) * Radius, np.cos(angle) * Radius, height])
        allcfs.crazyfliesById[cfid].goTo(pos, 0, 3.0)

    timeHelper.sleep(3.5)

    allcfs.setParam("usd/logging", 1)

    # disable ground effect in the NN computation
    allcfs.setParam("ctrlFa/enableGround", 0)

    for useNN in [0, 2]:
    # for useNN in [2]:
        # timeHelper.sleep(2.0)
        allcfs.setParam("ctrlFa/enableNN", useNN)

        for swapTime in SwapTimes:
            # swap 1
            for angle, cfid, height in zip(angles[len(Ids):], Ids, Heights):
                pos = np.array([np.sin(angle) * Radius, np.cos(angle) * Radius, height])
                allcfs.crazyfliesById[cfid].goTo(pos, 0, 3.0)
            timeHelper.sleep(swapTime+1.5)

            # swap 2
            for angle, cfid, height in zip(angles[0:len(Ids)], Ids, Heights):
                pos = np.array([np.sin(angle) * Radius, np.cos(angle) * Radius, height])
                allcfs.crazyfliesById[cfid].goTo(pos, 0, 3.0)
            timeHelper.sleep(swapTime+1.5)

    allcfs.setParam("usd/logging", 0)

    for cf in allcfs.crazyflies:
        pos = np.array(cf.initialPosition) + np.array([0, 0, np.min(Heights)])
        cf.goTo(pos, 0, 3.0)
    timeHelper.sleep(3.5)

    allcfs.land(targetHeight=0.02, duration=3.0)
    timeHelper.sleep(3.0)

