# neural-swarm
Code related to the Neural-Swarm (ICRA 2020, Journal) papers

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
