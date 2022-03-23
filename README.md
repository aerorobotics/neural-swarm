# neural-swarm
Code and data related to the [Neural-Swarm](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9196800) and [Neural-Swarm2](https://ieeexplore.ieee.org/abstract/document/9508420) (ICRA 2020, T-RO 2021) papers

For training, please run `learning/training.py`, which will use data in `data/training`.

Note that `learning/training.py` also includes data preprocessing, model saving, analysis, and visualization (please see comments therein).

This relies on some data files from Box, to be copied to the data/ folder.
This process can be automated using `rclone`:

- Setup rclone with box, following https://rclone.org/box/ (note "user" type worked just fine for me)
- To pull the latest data use: 

```
rclone copy box:neural-swarm data/ --dry-run
rclone copy box:neural-swarm data/ -P
```
- To push the latest data use: 

```
rclone copy data/ box:neural-swarm --dry-run
rclone copy data/ box:neural-swarm -P
```
