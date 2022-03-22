Robots:
  * CF50, CF51, CF52: Crazyflie 2.1 (i.e., new IMU), unmodified, use defaultcf.txt for system id.
    * ground plane: 0.05m
  * CF101, CF102: Crazyflie 2.1 (i.e., new IMU), on parrot frame, use parrot_frame.txt for system id. 
    * ground plane: 0.06m
  * CF100: Crazyflie 2.0 (old IMU), on parrot frame, use parrot_frame for system id.

Experiments:
  * NN enabled (epoch20_lip3_h20_f0d35)

  * swap_LLL: 
    * some ground touching
    * WARNING: DO NOT USE CF100 for FA ESTIMATION

  * randomwalk_lll:
    * Need to manually find end time for each of them (crashes on every flight!)
    * WARNING: DO NOT USE CF100 for FA ESTIMATION