import math
import os
import signal
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleOdometry
)


class OffboardPentagramBagNode(Node):
    """Offboard controller that flies a pentagram and records ROS bag data."""

    def __init__(self) -> None:
        super().__init__('offboard_pentagram_bag_node')

        self.get_logger().info("Offboard Pentagram Bag Node Alive!")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile
        )
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos_profile
        )
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile
        )

        # Subscribers
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile
        )
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile
        )
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback,
            qos_profile
        )

        # Mission parameters
        self.rate_hz = 20.0
        self.timer_period = 0.1
        self.altitude = -1.5
        self.star_radius = 1.5
        self.edge_traverse_time_s = 1.6
        self.corner_hold_time_s = 0.5
        self.takeoff_hold_time_s = 10.0
        self.final_hold_time_s = 2.0

        self.edge_steps = int(self.edge_traverse_time_s * self.rate_hz)
        self.corner_hold_steps = int(self.corner_hold_time_s * self.rate_hz)
        self.final_hold_steps = int(self.final_hold_time_s * self.rate_hz)

        self.vertices = self.compute_outer_vertices()

        # Proper pentagram traversal order
        self.star_order = [0, 2, 4, 1, 3, 0]

        self.path = []
        self.offboard_arr_counter = 0
        self.offboard_setpoint_counter = 0
        self.mission_started = False
        self.mission_finished = False
        self.armed = False

        self.start_time = time.time()

        self.vehicle_status = VehicleStatus()
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_odometry = VehicleOdometry()

        self.bag_process = None
        self.bag_started = False
        self.bag_output_dir = f"pentagram_bag_{int(time.time())}"

        self.init_pentagram_path()
        self.start_rosbag_recording()

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def compute_outer_vertices(self):
        """Compute the 5 outer vertices of a regular pentagon centered at the origin."""
        vertices = []
        for i in range(5):
            angle_deg = 90.0 - i * 72.0
            angle_rad = math.radians(angle_deg)
            x = self.star_radius * math.cos(angle_rad)
            y = self.star_radius * math.sin(angle_rad)
            vertices.append([x, y, self.altitude])
        return vertices

    def init_pentagram_path(self) -> None:
        """Build the pentagram path by connecting every second outer vertex."""
        self.path = []

        self.get_logger().info("Pentagram vertices:")
        for idx, v in enumerate(self.vertices, start=1):
            self.get_logger().info(
                f"V{idx}: x={v[0]:.3f}, y={v[1]:.3f}, z={v[2]:.3f}"
            )

        self.get_logger().info("Pentagram traversal order:")
        self.get_logger().info(" -> ".join([f"V{i + 1}" for i in self.star_order]))

        for i in range(len(self.star_order) - 1):
            start = self.vertices[self.star_order[i]]
            end = self.vertices[self.star_order[i + 1]]

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            yaw = math.atan2(dy, dx)

            # Hold at each star vertex
            for _ in range(self.corner_hold_steps):
                msg = TrajectorySetpoint()
                msg.position = [start[0], start[1], start[2]]
                msg.velocity = [0.0, 0.0, 0.0]
                msg.acceleration = [0.0, 0.0, 0.0]
                msg.yaw = yaw
                msg.yawspeed = 0.0
                self.path.append(msg)

            # Traverse the edge
            for step in range(1, self.edge_steps + 1):
                alpha = step / self.edge_steps
                x = start[0] + alpha * dx
                y = start[1] + alpha * dy
                z = self.altitude

                msg = TrajectorySetpoint()
                msg.position = [x, y, z]
                msg.velocity = [0.0, 0.0, 0.0]
                msg.acceleration = [0.0, 0.0, 0.0]
                msg.yaw = yaw
                msg.yawspeed = 0.0
                self.path.append(msg)

    def start_rosbag_recording(self) -> None:
        """Start rosbag recording in a separate process."""
        if self.bag_started:
            return

        topics_to_record = [
            '/fmu/out/vehicle_odometry',
            '/fmu/out/vehicle_local_position',
            '/fmu/out/vehicle_status',
            '/fmu/in/trajectory_setpoint',
            '/fmu/in/offboard_control_mode',
        ]

        command = ['ros2', 'bag', 'record', '-o', self.bag_output_dir] + topics_to_record

        try:
            self.bag_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            self.bag_started = True
            self.get_logger().info(f"Started rosbag recording: {self.bag_output_dir}")
        except Exception as exc:
            self.get_logger().error(f"Failed to start rosbag recording: {exc}")

    def stop_rosbag_recording(self) -> None:
        """Stop rosbag recording cleanly."""
        if self.bag_process is None:
            return

        try:
            os.killpg(os.getpgid(self.bag_process.pid), signal.SIGINT)
            self.bag_process.wait(timeout=10.0)
            self.get_logger().info(f"Stopped rosbag recording: {self.bag_output_dir}")
        except Exception as exc:
            self.get_logger().error(f"Failed to stop rosbag cleanly: {exc}")
            try:
                os.killpg(os.getpgid(self.bag_process.pid), signal.SIGTERM)
            except Exception:
                pass
        finally:
            self.bag_process = None

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition) -> None:
        self.vehicle_local_position = msg

    def vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        self.vehicle_odometry = msg

    def timer_callback(self) -> None:
        """Main mission execution loop."""
        self.publish_offboard_control_heartbeat_signal()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.armed = True

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        elapsed = time.time() - self.start_time

        if elapsed < self.takeoff_hold_time_s:
            self.publish_hold_setpoint(0.0, 0.0, self.altitude, yaw=0.0)
            return

        if not self.mission_started:
            self.mission_started = True
            self.get_logger().info("Tracing pentagram now")

        if self.offboard_arr_counter < len(self.path):
            msg = self.path[self.offboard_arr_counter]
            msg.timestamp = self.timestamp_us()
            self.trajectory_setpoint_publisher.publish(msg)
            self.offboard_arr_counter += 1
            return

        if self.offboard_arr_counter < len(self.path) + self.final_hold_steps:
            self.publish_hold_setpoint(0.0, 0.0, self.altitude, yaw=0.0)
            self.offboard_arr_counter += 1
            return

        if not self.mission_finished:
            self.mission_finished = True
            self.land()
            self.stop_rosbag_recording()

    def publish_hold_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.acceleration = [0.0, 0.0, 0.0]
        msg.yaw = yaw
        msg.yawspeed = 0.0
        msg.timestamp = self.timestamp_us()
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_offboard_control_heartbeat_signal(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self.timestamp_us()
        self.offboard_control_mode_publisher.publish(msg)

    def arm(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0
        )
        self.get_logger().info('Arm command sent')

    def disarm(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0
        )
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0
        )
        self.get_logger().info("Switching to offboard mode")

    def land(self) -> None:
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_vehicle_command(self, command: int, **params) -> None:
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.timestamp_us()
        self.vehicle_command_publisher.publish(msg)

    def timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def destroy_node(self):
        """Ensure bagging stops even during interruption."""
        self.stop_rosbag_recording()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OffboardPentagramBagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
