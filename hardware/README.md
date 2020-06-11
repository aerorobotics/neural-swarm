# Hardware Experiments for Neural-Swarm

## ROS Integration

### Initial Setup

* Checkout crazyswarm
* Use symbolic link to include `neural-swarm-ros-pkg`: 

```
crazyswarm/ros_ws/src/userPackages$ ln -s /path/to/neural-swarm-ros-pkg .
```

### Run

```
crazyswarm/ros_ws/crazyswarm/scripts$ python chooser.py --basepath ../../userPackages/neural-swarm-ros-pkg/launch/ --stm32Fw ~/projects/caltech/neural-swarm/hardware/crazyflie-firmware/cf2.bin
$ roslaunch neural-swarm neural-swarm.launch
neural-swarm-ros-pkg/scripts$ export PYTHONPATH=$PYTHONPATH:/path/to/crazyswarm/ros_ws/src/crazyswarm/scripts
neural-swarm-ros-pkg/scripts$ examplescript.py
```
