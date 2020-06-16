#!/usr/bin/env python

import numpy as np
import threading
import time

from pycrazyswarm import *

if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.setParam("usd/logging", 1)

    # for useNN in [0, 2]:
    for useNN in [2]:
        allcfs.setParam("ctrlFa/enableNN", useNN)

        for duration in [1.5,2.5,3.5]:
            allcfs.takeoff(targetHeight=0.5, duration=duration)
            timeHelper.sleep(duration + 0.5)

            allcfs.land(targetHeight=0.02, duration=duration)
            timeHelper.sleep(duration + 0.5)

    allcfs.setParam("usd/logging", 0)



