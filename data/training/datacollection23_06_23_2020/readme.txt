Robots:
  * CF50, CF51, CF52: Crazyflie 2.1 (i.e., new IMU), unmodified, use defaultcf.txt for system id.
    * ground plane: 0.05m
  * CF101, CF102: Crazyflie 2.1 (i.e., new IMU), on parrot frame, use parrot_frame.txt for system id. 
    * ground plane: 0.06m
  * CF100: Crazyflie 2.0 (old IMU), on parrot frame, use parrot_frame for system id.

Uses val_with22/epoch40_lip3_h20_f0d35_B256 (GE enabled)


randomwalk_lll: most of them crashed after just a few seconds of flight (because of CF100 bad IMU). Might not be worth the effort using?