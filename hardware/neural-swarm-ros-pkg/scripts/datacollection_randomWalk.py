#!/usr/bin/env python

import numpy as np
import threading
import time

from pycrazyswarm import *
import uav_trajectory

# # for 2-robot data
# Offset = np.array([0.5,0,0.9])
# DX = 0.2
# DY = 0.2
# DZ = 0.6

# for 3-robot data
Offset = np.array([0.5,0,0.9])
DX = 0.3
DY = 0.3
DZ = 0.6

# # for ground effect data:
# Offset = np.array([0.5,0,0.23])
# DX = 0.2
# DY = 0.2
# DZ = 0.36


SPEED = 0.3 #m/s
DURATION = 60

def threadsafeSleep(duration):
    startTime = timeHelper.time()
    while timeHelper.time() - startTime < duration:
        time.sleep(0.1)

def thread_func(cf, duration):
    pos = cf.position()
    startTime = timeHelper.time()
    while timeHelper.time() - startTime < duration:
        newPos = np.array([
            np.random.uniform(-DX/2, DX/2),
            np.random.uniform(-DY/2, DY/2),
            np.random.uniform(-DZ/2, DZ/2)]) + Offset
        yaw = 0 #np.random.uniform(0, 2*math.pi)
        dist = np.linalg.norm(pos - newPos)
        t = max(dist/SPEED, 2.0)
        # print(newPos, yaw, t)
        cf.goTo(newPos, yaw, t)
        pos = newPos

        threadsafeSleep(t/2)
        # timeHelper.sleep(t)


if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    allcfs.takeoff(targetHeight=Offset[2], duration=3.5)
    timeHelper.sleep(3.5)

    allcfs.setParam("ctrlFa/enableGround", 1) # enable ground effect computation
    allcfs.setParam("ctrlFa/enableNN", 2) # enable NN

    allcfs.setParam("planner/enAP", 1)

    allcfs.setParam("usd/logging", 1)

    threads = []
    for cf in allcfs.crazyflies:
        t = threading.Thread(target=thread_func, args=(
        cf,
        DURATION))
        t.start()
        threads.append(t)

    while True:
        allThreadsDone = True
        for t in threads:
            if t.is_alive():
                allThreadsDone = False
                break
        if allThreadsDone:
            break
        timeHelper.sleep(0.1)

    for cf in allcfs.crazyflies:
        pos = np.array(cf.initialPosition) + np.array([0, 0, Offset[2]])
        cf.goTo(pos, 0, 3.0)
    timeHelper.sleep(3.5)

    allcfs.setParam("usd/logging", 0)
    allcfs.setParam("planner/enAP", 0)

    # allcfs.setParam("planner/enableAP", 0)

    allcfs.land(targetHeight=0.02, duration=3.5)
    timeHelper.sleep(3.5)

