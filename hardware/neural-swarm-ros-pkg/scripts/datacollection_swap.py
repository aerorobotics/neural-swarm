#!/usr/bin/env python

import numpy as np
import random

from pycrazyswarm import *
import uav_trajectory

Min_Height = 0.5
# Min_Height = 0.7 # use this for LLL

SwapTimes = [4,3,2]

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # randomly shuffle enabled IDs
    ids_set = set(allcfs.crazyfliesById.keys())

    # If we have an old CF 2.0, make sure it is always on top
    if 100 in ids_set:
        ids_set.remove(100)
        Ids = list(ids_set)
        random.shuffle(Ids)
        Ids.append(100)
    else:
        Ids = list(ids_set)
        random.shuffle(Ids)

    Heights = [Min_Height]
    for i in range(len(Ids)-1):
        if Ids[i] < 100 and Ids[i+1] < 100:
            # both are small CFs
            Heights.append(Heights[i] + np.random.uniform(0.2, 0.4))
        elif Ids[i] < 100 and Ids[i+1] >= 100:
            # below is small, top is large
            Heights.append(Heights[i] + np.random.uniform(0.4, 0.6))
        elif Ids[i] >= 100 and Ids[i+1] < 100:
            # below is large, top is small
            Heights.append(Heights[i] + np.random.uniform(0.1, 0.3))
        elif Ids[i] >= 100 and Ids[i+1] >= 100:
            # both are large
            Heights.append(Heights[i] + np.random.uniform(0.2, 0.4))

    Radius = 0.25

    # randomly sample initial angles
    angles = np.random.uniform(0, 2*np.pi, len(Ids))

    allcfs.takeoff(targetHeight=np.min(Heights), duration=3.0)
    timeHelper.sleep(3.5)

    # go to initial positions
    for angle, cfid, height in zip(angles[0:len(Ids)], Ids, Heights):
        pos = np.array([np.cos(angle) * Radius, np.sin(angle) * Radius, height])
        allcfs.crazyfliesById[cfid].goTo(pos, 0, 3.0)

    timeHelper.sleep(3.5)

    allcfs.setParam("usd/logging", 1)

    # enable ground effect in the NN computation
    allcfs.setParam("ctrlFa/enableGround", 1)

    # for useNN in [0, 2]:
    for useNN in [2]:
        # timeHelper.sleep(2.0)
        allcfs.setParam("ctrlFa/enableNN", useNN)

        for swapTime in SwapTimes:
            # swap 1
            for angle, cfid, height in zip(angles, Ids, Heights):
                pos = np.array([np.cos(angle + np.pi) * Radius, np.sin(angle + np.pi) * Radius, height])
                allcfs.crazyfliesById[cfid].goTo(pos, 0, swapTime)
            timeHelper.sleep(swapTime+0.1)

            # swap 2
            for angle, cfid, height in zip(angles, Ids, Heights):
                pos = np.array([np.cos(angle) * Radius, np.sin(angle) * Radius, height])
                allcfs.crazyfliesById[cfid].goTo(pos, 0, swapTime)
            timeHelper.sleep(swapTime+0.1)

    allcfs.setParam("usd/logging", 0)

    for cf in allcfs.crazyflies:
        pos = np.array(cf.initialPosition) + np.array([0, 0, np.min(Heights)])
        cf.goTo(pos, 0, 3.0)
    timeHelper.sleep(3.5)

    allcfs.land(targetHeight=0.02, duration=3.0)
    timeHelper.sleep(3.0)

