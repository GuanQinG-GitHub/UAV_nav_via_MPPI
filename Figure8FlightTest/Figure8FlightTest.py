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


class OffboardFigure8BagNode(Node):
    """Offboard controller that flies a smooth figure-8 and records ROS bag data."""

    def __init__(self) -> None:
        super().__init__('offboard_figure8_bag_node')

        self.get_logger().info("Offboard Figure-8 Bag Node Alive!")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

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
        self.timer_period = 1.0 / self.rate_hz

        self.altitude = -1.5

        # Figure-8 shape parameters
        self.x_amplitude = 1.8
        self.y_amplitude = 1.0
        self.figure8_period_s = 14.0
        self.num_cycles = 2

        self.takeoff_hold_time_s = 10.0
        self.final_hold_time_s = 3.0

        self.figure8_steps = int(self.figure8_period_s * self.rate_hz * self.num_cycles)
        self.final_hold_steps = int(self.final_hold_time_s * self.rate_hz)

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
        self.bag_output_dir = f"figure8_bag_{int(time.time())}"

        self.init_figure8_path()
        self.start_rosbag_recording()

        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def init_figure8_path(self) -> None:
        """
        Build a smooth horizontal figure-8 path.

        Parametric curve:
            x(t) = A * sin(t)
            y(t) = B * sin(2t)

        This creates a sideways figure-8 centered at the origin.
        """
        self.path = []

        self.get_logger().info("Building figure-8 trajectory")
        self.get_logger().info(
            f"x_amplitude={self.x_amplitude:.2f}, "
            f"y_amplitude={self.y_amplitude:.2f}, "
            f"z={self.altitude:.2f}"
        )

        total_angle = 2.0 * math.pi * self.num_cycles

        previous_x = 0.0
        previous_y = 0.0

        for step in range(self.figure8_steps + 1):
            theta = total_angle * step / self.figure8_steps

            x = self.x_amplitude * math.sin(theta)
            y = self.y_amplitude * math.sin(2.0 * theta)
            z = self.altitude

            if step == 0:
                yaw = 0.0
            else:
                dx = x - previous_x
                dy = y - previous_y
                yaw = math.atan2(dy, dx)

            msg = TrajectorySetpoint()
            msg.position = [x, y, z]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.acceleration = [0.0, 0.0, 0.0]
            msg.yaw = yaw
            msg.yawspeed = 0.0

            self.path.append(msg)

            previous_x = x
            previous_y = y

        self.get_logger().info(f"Generated {len(self.path)} figure-8 setpoints")

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
            self.get_logger().info("Tracing figure-8 now")

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
    node = OffboardFigure8BagNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()