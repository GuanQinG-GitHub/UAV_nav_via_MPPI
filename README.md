# UAV_nav_via_MPPI

This repository collects the code, workspace snapshots, and analysis scripts for the ECE 591 Software for Robotics project on UAV navigation with PX4, ROS 2, and MPPI-style trajectory tracking.

At a high level, the repo combines:

- offboard flight-control experiments for hovering and reference tracking
- shape-flight data-collection scripts for square and pentagram trajectories
- an MPPI-based figure-eight tracking prototype
- local 3D obstacle-perception utilities for cropped point clouds and obstacle metrics
- ROS 2 bag analysis utilities for comparing commanded and measured motion
- archived ROS 2 workspace snapshots used during development

This is currently organized as a working project repository rather than a polished release package, so it includes both active scripts and intermediate workspace snapshots.

## High-Level Purpose

The project goal is to explore how a UAV can be commanded in PX4 offboard mode with increasingly capable reference generators:

1. start from simple hovering and shape-tracing tests
2. move toward richer trajectory tracking with MPPI-style optimization
3. add local perception utilities that can support obstacle-aware navigation
4. analyze recorded ROS 2 bag data to evaluate tracking behavior

## Repository Organization

```text
UAV_nav_via_MPPI/
├── .gitignore
├── 3DVoxelMap_ws/
│   └── src/
│       └── starling_local_world_model/
│           ├── CMakeLists.txt
│           ├── package.xml
│           └── src/
│               ├── local_cloud_cropper.cpp
│               └── local_obstacle_metrics.cpp
├── CPU_parellel_test/
│   ├── benchmark_parallel.py
│   ├── benchmark_serial.py
│   └── benchmark_vecorized.py
├── HoveringTest/
│   └── hover.py
├── PentagramFlightTest/
│   └── PentagramFlightTest.py
├── SquareFlightTest/
│   └── SquareFlightTest.py
├── analyze_shape/
│   ├── analyze_shape_bag.py
│   └── analyze_shape_bag.py.save
├── april_18_ws/
│   └── hover.py
├── march_31_ws/
│   ├── build/
│   ├── install/
│   ├── log/
│   └── src/
│       ├── px4_msgs/
│       ├── px4_ros_com/
│       └── starling_hover/
├── temp_mppi/
│   └── figure8_mppi_node.py
└── README.md
```

## Directory Guide

### `3DVoxelMap_ws`

Experimental ROS 2 perception workspace content for local obstacle awareness.

- `local_cloud_cropper.cpp` crops a global/aligned point cloud around the current UAV pose and republishes a local cloud.
- `local_obstacle_metrics.cpp` computes simple distance metrics from the local cloud, such as nearest overall, front, left, right, up, and down obstacle distances.

### `CPU_parellel_test`

Micro-benchmarks for rollout computation strategies that are relevant to MPPI-style control.

- `benchmark_serial.py` evaluates a serial rollout loop.
- `benchmark_vecorized.py` evaluates a NumPy-vectorized version.
- `benchmark_parallel.py` evaluates multiprocessing-based rollouts.

These scripts are useful for understanding the computational tradeoffs behind sampling-based control.

### `HoveringTest`

Early PX4 offboard test script for basic hover behavior and simple position commands. This is one of the simplest controller prototypes in the repo.

### `PentagramFlightTest`

ROS 2 offboard flight script that commands a UAV to trace a pentagram trajectory and records ROS bag data for later analysis.

### `SquareFlightTest`

ROS 2 offboard flight script that commands a UAV to trace a square trajectory while recording reference and vehicle-state data to rosbag.

### `analyze_shape`

Offline analysis utilities for recorded ROS 2 bag files.

- `analyze_shape_bag.py` reads commanded setpoints and measured PX4 state topics from a bag, converts them into aligned arrays, and supports plotting / comparison workflows.
- `analyze_shape_bag.py.save` is a backup copy of the same script.

### `april_18_ws`

A later standalone hover / offboard experiment script. It is similar in spirit to `HoveringTest`, but represents a separate iteration of controller testing.

### `march_31_ws`

Archived ROS 2 workspace snapshot from an earlier development stage.

- `src/px4_msgs/` contains PX4 ROS message definitions.
- `src/px4_ros_com/` contains PX4 ROS communication examples and utilities.
- `src/starling_hover/` contains a custom hover package.
- `build/`, `install/`, and `log/` contain generated colcon artifacts that were preserved in this snapshot.

This directory is best understood as a captured workspace state, not a minimal source-only package.

### `temp_mppi`

Contains the current MPPI-oriented prototype:

- `figure8_mppi_node.py` is a ROS 2 node that extends a PX4 figure-eight offboard example with a high-level MPPI tracker. PX4 still handles low-level control, while this node generates optimized short-horizon reference updates.

## Workflow View

The repository roughly breaks down into this progression:

- baseline offboard control: `HoveringTest`, `april_18_ws`
- structured flight experiments: `SquareFlightTest`, `PentagramFlightTest`
- bag-based evaluation: `analyze_shape`
- sampling-based control prototype: `temp_mppi`
- perception support for obstacle-aware navigation: `3DVoxelMap_ws`
- archived development environment snapshot: `march_31_ws`

## Notes

- The repository currently mixes source code, recorded-workflow utilities, and workspace snapshots.
- Some directories are experimental and may depend on a specific PX4, ROS 2, and Starling/VOXL setup.
- `march_31_ws` includes generated artifacts, so it is much larger and less tidy than the source-only folders.
