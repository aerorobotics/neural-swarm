Robots:
  * CF50, CF51, CF52: Crazyflie 2.1 (i.e., new IMU), unmodified, use defaultcf.txt for system id.
    * ground plane: 0.05m
  * CF101, CF102: Crazyflie 2.1 (i.e., new IMU), on parrot frame, use parrot_frame.txt for system id. 
    * ground plane: 0.06m
  * CF100: Crazyflie 2.0 (old IMU), on parrot frame, use parrot_frame for system id.

Experiments:
  1. swapSS [cf50, cf51]
    * datacollection_swap.py (every run has random dist in [0.2,...,0.4], and 3 swap times)
    * NN enabled w/o ground: val_filter/epoch20_lip3_h20_f0d4

  2. swapSSS [cf50, cf51, cf52]
    * see swapSS

  3. swapLL [cf101, cf102]
    * see swapSS

  4. swapSL
    * see swapSS

  5. swapSSL
    * see swapSS

  6. swapLLL
    * WARNING: DO NOT USE FA ESTIMATE FROM CF100 (which is CF 2.0 not 2.1, i.e., much worse IMU)

  7. randomWalkLL
    * WARNING: flights 00,01,02, 04 touched the ground -> needs filtering

  8. randomWalkSLL
    * WARNING: some flights had short ground touches -> needs filtering

  9. randomWalkSSL
    * WARNING: one flight had short ground touches -> needs filtering

  10. takeoffS
    * run with different drones
    * NN ground effect enabled

  11. randomWalkS (single small CF close to the ground)
    * a few light touches -> needs filtering
    * NN ground effect enabled

  12. randomWalkL (single large CF close to the ground)
    * a few light touches -> needs filtering
    * NN ground effect enabled