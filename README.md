# Filters

## Quickstart

### Clustering

```bash
ros2 run bb_filters cluster_poses_node.py --ros-args -p input_pose_topic:=/auv4/front_cam/image_matching/pose -p num_poses:=100
```

## Notes

### Cluster Poses vs Transforms

We implement a separate `cluster_poses_node` to:

- **Control time alignment explicitly**. We can ensure each detected pose is paired with an odometry sample within a chosen tolerance, rather than relying on latest available transforms.
- **Produce physically meaningful clustering inputs**. Each data point corresponds to a real detection event transformed using the odometry state at approximately the same time, rather than "virtual" points created by mixing a detection at time t1 with odometry at time t2. (When clustering transforms, you may feed in multiple copies of the same detection transformed with different odometry readings.)
- **Achieve higher efficiency** from not having to subscribe to a (potentially) high frequency dynamic transforms topic `/tf` and not having to use `tf` lookups. (But this is balanced out by the overhead from managing the time synchronization ourselves instead of using a `tf2_ros.Buffer`.)


### Cluster Poses implementation considerations

- **TOCTOU race condition** see [here](https://github.com/ros2/rclpy/issues/1206)
- **Performance drawback with MultithreadedExecutors** see [here](https://github.com/ros2/rclpy/issues/1452)
