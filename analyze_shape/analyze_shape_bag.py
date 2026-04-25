#!/usr/bin/env python3

import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TRAJECTORY_TOPIC = "/fmu/in/trajectory_setpoint"
ODOMETRY_TOPIC = "/fmu/out/vehicle_odometry"
LOCAL_POSITION_TOPIC = "/fmu/out/vehicle_local_position"


def safe_make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_topic_type_map(bag_path: str):
    """
    Open bag metadata and return a mapping:
        topic_name -> topic_type_string
    """
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    return {topic.name: topic.type for topic in topic_types}


def extract_time_seconds(msg, bag_timestamp_ns: int) -> float:
    """
    Prefer PX4 message timestamp if available.
    PX4 timestamps are typically in microseconds.
    Fall back to rosbag timestamp in nanoseconds.
    """
    if hasattr(msg, "timestamp"):
        ts = getattr(msg, "timestamp")
        if ts is not None and ts != 0:
            return float(ts) * 1e-6
    return float(bag_timestamp_ns) * 1e-9


def read_bag_messages(bag_path: str, requested_topics):
    """
    Read messages from a ROS 2 bag and return a dict:
        topic -> list of (time_s, msg)
    """
    topic_type_map = get_topic_type_map(bag_path)

    missing = [topic for topic in requested_topics if topic not in topic_type_map]
    if missing:
        print("Warning: These requested topics were not found in the bag:")
        for topic in missing:
            print(f"  {topic}")

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    data = {topic: [] for topic in requested_topics}

    while reader.has_next():
        topic_name, serialized_data, bag_timestamp_ns = reader.read_next()

        if topic_name not in requested_topics:
            continue
        if topic_name not in topic_type_map:
            continue

        msg_type = get_message(topic_type_map[topic_name])
        msg = deserialize_message(serialized_data, msg_type)
        time_s = extract_time_seconds(msg, bag_timestamp_ns)
        data[topic_name].append((time_s, msg))

    return data


def trajectory_setpoints_to_arrays(messages):
    """
    Convert /fmu/in/trajectory_setpoint messages into arrays.

    Expected PX4 field usage:
        msg.position[0] = x
        msg.position[1] = y
        msg.position[2] = z
        msg.yaw = yaw
    """
    t = []
    x = []
    y = []
    z = []
    yaw = []

    for time_s, msg in messages:
        if not hasattr(msg, "position"):
            continue
        pos = msg.position
        if len(pos) < 3:
            continue

        t.append(time_s)
        x.append(float(pos[0]))
        y.append(float(pos[1]))
        z.append(float(pos[2]))
        yaw.append(float(getattr(msg, "yaw", 0.0)))

    return {
        "t": np.array(t, dtype=float),
        "x": np.array(x, dtype=float),
        "y": np.array(y, dtype=float),
        "z": np.array(z, dtype=float),
        "yaw": np.array(yaw, dtype=float),
    }


def vehicle_odometry_to_arrays(messages):
    """
    Convert /fmu/out/vehicle_odometry messages into arrays.

    PX4 VehicleOdometry typically stores:
        msg.position[0] = x
        msg.position[1] = y
        msg.position[2] = z
    """
    t = []
    x = []
    y = []
    z = []

    for time_s, msg in messages:
        if not hasattr(msg, "position"):
            continue
        pos = msg.position
        if len(pos) < 3:
            continue

        t.append(time_s)
        x.append(float(pos[0]))
        y.append(float(pos[1]))
        z.append(float(pos[2]))

    return {
        "t": np.array(t, dtype=float),
        "x": np.array(x, dtype=float),
        "y": np.array(y, dtype=float),
        "z": np.array(z, dtype=float),
    }


def vehicle_local_position_to_arrays(messages):
    """
    Convert /fmu/out/vehicle_local_position messages into arrays.

    PX4 VehicleLocalPosition typically stores:
        msg.x, msg.y, msg.z
    """
    t = []
    x = []
    y = []
    z = []

    for time_s, msg in messages:
        if not (hasattr(msg, "x") and hasattr(msg, "y") and hasattr(msg, "z")):
            continue

        t.append(time_s)
        x.append(float(msg.x))
        y.append(float(msg.y))
        z.append(float(msg.z))

    return {
        "t": np.array(t, dtype=float),
        "x": np.array(x, dtype=float),
        "y": np.array(y, dtype=float),
        "z": np.array(z, dtype=float),
    }


def sort_by_time(data_dict):
    """
    Sort arrays by time.
    """
    if data_dict["t"].size == 0:
        return data_dict

    order = np.argsort(data_dict["t"])
    return {k: v[order] for k, v in data_dict.items()}


def normalize_timebase(data_dict, t0: float):
    """
    Shift time so the reference time starts at 0.
    """
    out = dict(data_dict)
    out["t"] = out["t"] - t0
    return out


def trim_to_overlap(cmd, actual):
    """
    Trim both datasets to their overlapping time interval.
    """
    start_t = max(cmd["t"][0], actual["t"][0])
    end_t = min(cmd["t"][-1], actual["t"][-1])

    if end_t <= start_t:
        raise RuntimeError(
            "No overlapping time interval found between commanded and actual data."
        )

    cmd_mask = (cmd["t"] >= start_t) & (cmd["t"] <= end_t)
    actual_mask = (actual["t"] >= start_t) & (actual["t"] <= end_t)

    cmd_trimmed = {k: v[cmd_mask] for k, v in cmd.items()}
    actual_trimmed = {k: v[actual_mask] for k, v in actual.items()}

    return cmd_trimmed, actual_trimmed


def interpolate_actual_onto_command_time(cmd, actual):
    """
    Interpolate actual x, y, z onto the command timestamps.
    This gives us samples at the same times so error(t) is meaningful.
    """
    t_cmd = cmd["t"]
    t_actual = actual["t"]

    x_actual_interp = np.interp(t_cmd, t_actual, actual["x"])
    y_actual_interp = np.interp(t_cmd, t_actual, actual["y"])
    z_actual_interp = np.interp(t_cmd, t_actual, actual["z"])

    return {
        "t": t_cmd.copy(),
        "x": x_actual_interp,
        "y": y_actual_interp,
        "z": z_actual_interp,
    }


def compute_xy_error(cmd, actual_interp):
    """
    Compute exactly:
        error(t) = sqrt((x_cmd - x_actual)^2 + (y_cmd - y_actual)^2)
    """
    dx = cmd["x"] - actual_interp["x"]
    dy = cmd["y"] - actual_interp["y"]
    error = np.sqrt(dx**2 + dy**2)
    return error


def summarize_error(error_xy, cmd, actual_interp):
    """
    Compute useful summary metrics.
    """
    rmse = float(np.sqrt(np.mean(error_xy**2)))
    mean_err = float(np.mean(error_xy))
    max_err = float(np.max(error_xy))
    min_err = float(np.min(error_xy))

    z_error = cmd["z"] - actual_interp["z"]
    z_rmse = float(np.sqrt(np.mean(z_error**2)))
    z_mean_abs = float(np.mean(np.abs(z_error)))

    return {
        "samples": int(error_xy.size),
        "xy_rmse_m": rmse,
        "xy_mean_error_m": mean_err,
        "xy_max_error_m": max_err,
        "xy_min_error_m": min_err,
        "z_rmse_m": z_rmse,
        "z_mean_abs_error_m": z_mean_abs,
    }


def save_xy_overlay_plot(cmd, actual_interp, output_dir: Path, title_suffix: str):
    plt.figure(figsize=(8, 8))
    plt.plot(cmd["x"], cmd["y"], label="Commanded Path")
    plt.plot(actual_interp["x"], actual_interp["y"], label="Actual Path")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"XY Path Overlay{title_suffix}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "xy_overlay.png", dpi=300)
    plt.close()


def save_z_vs_time_plot(cmd, actual_interp, output_dir: Path, title_suffix: str):
    plt.figure(figsize=(10, 5))
    plt.plot(cmd["t"], cmd["z"], label="Commanded Z")
    plt.plot(actual_interp["t"], actual_interp["z"], label="Actual Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Z (m)")
    plt.title(f"Z vs Time{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "z_vs_time.png", dpi=300)
    plt.close()


def save_error_plot(t, error_xy, output_dir: Path, title_suffix: str):
    plt.figure(figsize=(10, 5))
    plt.plot(t, error_xy, label="XY Tracking Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (m)")
    plt.title(f"XY Tracking Error vs Time{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "xy_error_vs_time.png", dpi=300)
    plt.close()


def save_summary_text(summary, output_dir: Path, actual_source_name: str):
    lines = [
        "Flight Tracking Summary",
        "=======================",
        f"Actual source: {actual_source_name}",
        f"Samples: {summary['samples']}",
        f"XY RMSE (m): {summary['xy_rmse_m']:.6f}",
        f"XY Mean Error (m): {summary['xy_mean_error_m']:.6f}",
        f"XY Max Error (m): {summary['xy_max_error_m']:.6f}",
        f"XY Min Error (m): {summary['xy_min_error_m']:.6f}",
        f"Z RMSE (m): {summary['z_rmse_m']:.6f}",
        f"Z Mean Abs Error (m): {summary['z_mean_abs_error_m']:.6f}",
    ]
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def choose_actual_source(raw_data, preferred_source: str):
    """
    Select actual-motion source according to user preference.
    """
    have_odom = len(raw_data.get(ODOMETRY_TOPIC, [])) > 0
    have_local = len(raw_data.get(LOCAL_POSITION_TOPIC, [])) > 0

    if preferred_source == "odometry":
        if not have_odom:
            raise RuntimeError(
                "Preferred actual source was odometry, but /fmu/out/vehicle_odometry "
                "was not found or had no messages."
            )
        return "odometry", vehicle_odometry_to_arrays(raw_data[ODOMETRY_TOPIC])

    if preferred_source == "local_position":
        if not have_local:
            raise RuntimeError(
                "Preferred actual source was local_position, but "
                "/fmu/out/vehicle_local_position was not found or had no messages."
            )
        return "local_position", vehicle_local_position_to_arrays(raw_data[LOCAL_POSITION_TOPIC])

    if preferred_source == "auto":
        if have_odom:
            return "odometry", vehicle_odometry_to_arrays(raw_data[ODOMETRY_TOPIC])
        if have_local:
            return "local_position", vehicle_local_position_to_arrays(raw_data[LOCAL_POSITION_TOPIC])

    raise RuntimeError(
        "Could not find a usable actual-motion topic in the bag. "
        "Expected /fmu/out/vehicle_odometry or /fmu/out/vehicle_local_position."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PX4 shape-tracing rosbag and plot commanded vs actual flight path."
    )
    parser.add_argument(
        "bag_path",
        help="Path to ROS 2 bag directory"
    )
    parser.add_argument(
        "--actual-source",
        choices=["auto", "odometry", "local_position"],
        default="auto",
        help="Which actual-motion topic to use"
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label appended to plot titles, example: ' - Square Flight'"
    )
    parser.add_argument(
        "--output-dir",
        default="shape_analysis_output",
        help="Directory to save plots and summary"
    )
    args = parser.parse_args()

    bag_path = args.bag_path
    output_dir = Path(args.output_dir)
    safe_make_dir(output_dir)

    if not os.path.isdir(bag_path):
        print(f"Error: bag path does not exist or is not a directory: {bag_path}")
        sys.exit(1)

    requested_topics = [TRAJECTORY_TOPIC, ODOMETRY_TOPIC, LOCAL_POSITION_TOPIC]

    print("Reading rosbag...")
    raw_data = read_bag_messages(bag_path, requested_topics)

    if len(raw_data.get(TRAJECTORY_TOPIC, [])) == 0:
        print(
            "Error: No commanded trajectory messages found on "
            f"{TRAJECTORY_TOPIC}"
        )
        sys.exit(1)

    cmd = trajectory_setpoints_to_arrays(raw_data[TRAJECTORY_TOPIC])
    cmd = sort_by_time(cmd)

    actual_source_name, actual = choose_actual_source(raw_data, args.actual_source)
    actual = sort_by_time(actual)

    if cmd["t"].size < 2:
        print("Error: Not enough commanded samples to analyze.")
        sys.exit(1)

    if actual["t"].size < 2:
        print("Error: Not enough actual-motion samples to analyze.")
        sys.exit(1)

    # Put both on a common absolute time frame, then trim overlap.
    cmd, actual = trim_to_overlap(cmd, actual)

    # Normalize time so overlap start is t=0 for nicer plots.
    t0 = min(cmd["t"][0], actual["t"][0])
    cmd = normalize_timebase(cmd, t0)
    actual = normalize_timebase(actual, t0)

    # Interpolate actual motion onto commanded timestamps.
    actual_interp = interpolate_actual_onto_command_time(cmd, actual)

    # Compute error exactly as requested.
    error_xy = compute_xy_error(cmd, actual_interp)

    # Summaries
    summary = summarize_error(error_xy, cmd, actual_interp)

    # Save outputs
    save_xy_overlay_plot(cmd, actual_interp, output_dir, args.label)
    save_z_vs_time_plot(cmd, actual_interp, output_dir, args.label)
    save_error_plot(cmd["t"], error_xy, output_dir, args.label)
    save_summary_text(summary, output_dir, actual_source_name)

    print("\nAnalysis complete.\n")
    print(f"Actual source used: {actual_source_name}")
    print(f"Output directory: {output_dir}")
    print(f"Samples: {summary['samples']}")
    print(f"XY RMSE (m): {summary['xy_rmse_m']:.6f}")
    print(f"XY Mean Error (m): {summary['xy_mean_error_m']:.6f}")
    print(f"XY Max Error (m): {summary['xy_max_error_m']:.6f}")
    print(f"Z RMSE (m): {summary['z_rmse_m']:.6f}")
    print("\nSaved files:")
    print(f"  {output_dir / 'xy_overlay.png'}")
    print(f"  {output_dir / 'z_vs_time.png'}")
    print(f"  {output_dir / 'xy_error_vs_time.png'}")
    print(f"  {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
