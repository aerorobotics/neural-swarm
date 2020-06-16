#!/usr/bin/env python

import numpy as np

from pycrazyswarm import *
import uav_trajectory

Id2 = 50 #102 #51
Id1 = 102#50 #101 #50
Pos1 = np.array([0.2, -0.2, 0.0])
Pos2 = np.array([0.2, 0.2, 0.0])
Height1 = 0.4
Height2 = 1.0


# SwapTimes = [4, 3, 2]

# SwapTimes = [20, 10]

SwapTimes = [4,2]
#SwapTimes = [3,3,3]

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    # allcfs.setParam("stabilizer/controller", 5) # use Lee controller

    allcfs.takeoff(targetHeight=Height1, duration=3.0)
    timeHelper.sleep(3.5)

    # go to initial positions
    allcfs.crazyfliesById[Id1].goTo(Pos1 + np.array([0,0,Height1]), 0, 3.0)
    allcfs.crazyfliesById[Id2].goTo(Pos2 + np.array([0,0,Height2]), 0, 3.0)
    timeHelper.sleep(3.5)
        
    allcfs.setParam("usd/logging", 1)

    for Height1 in [0.4, 0.6, 0.75]:
        allcfs.crazyfliesById[Id1].goTo(Pos1 + np.array([0,0,Height1]), 0, 1.5)
        timeHelper.sleep(2.0)
        for useNN in [0]:
            # timeHelper.sleep(2.0)
            # allcfs.setParam("ctrlSJC/resetCtrl", 1)
            allcfs.setParam("ctrlFa/enableNN", useNN)

            for swapTime in SwapTimes:
                # swap 1
                allcfs.crazyfliesById[Id1].goTo(Pos2 + np.array([0,0,Height1]), 0, swapTime)
                allcfs.crazyfliesById[Id2].goTo(Pos1 + np.array([0,0,Height2]), 0, swapTime)
                timeHelper.sleep(swapTime+1.5)

                # swap 2
                allcfs.crazyfliesById[Id1].goTo(Pos1 + np.array([0,0,Height1]), 0, swapTime)
                allcfs.crazyfliesById[Id2].goTo(Pos2 + np.array([0,0,Height2]), 0, swapTime)
                timeHelper.sleep(swapTime+1.5)

    allcfs.setParam("usd/logging", 0)

    for cf in allcfs.crazyflies:
        pos = np.array(cf.initialPosition) + np.array([0, 0, Height1])
        cf.goTo(pos, 0, 3.0)
    timeHelper.sleep(3.5)

    allcfs.land(targetHeight=0.02, duration=3.0)
    timeHelper.sleep(3.0)

