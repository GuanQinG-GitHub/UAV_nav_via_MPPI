import os
import math
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import rosbag2_py

from std_msgs.msg import Int32
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleOdometry,
)


class OffboardSquareBaggedNode(Node):
    """PX4 offboard node that traces a square and records odometry + references to rosbag."""

    def __init__(self) -> None:
        super().__init__('offboard_square_bagged_node')

        self.get_logger().info("Offboard Square Bagged Node Alive!")

        # =========================
        # QoS
        # =========================
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # =========================
        # Publishers
        # =========================
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Debug publishers for bagging / visualization
        self.reference_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/offboard_debug/reference', 10)
        self.active_vertex_publisher = self.create_publisher(
            Int32, '/offboard_debug/active_vertex', 10)

        # =========================
        # Subscribers
        # =========================
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

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

        # =========================
        # Motion parameters
        # =========================
        self.rate = 20.0                     # Hz
        self.altitude = -1.5                # NED z, meters
        self.side_length = 2.0              # meters
        self.half_side = self.side_length / 2.0
        self.edge_traverse_time = 1.6       # seconds per edge
        self.corner_hold_time = 0.5         # seconds hold at each vertex
        self.takeoff_hold_time = 10.0       # seconds at center before shape
        self.final_hold_time = 2.0          # seconds at center after shape

        self.edge_steps = max(1, int(self.edge_traverse_time * self.rate))
        self.corner_hold_steps = max(1, int(self.corner_hold_time * self.rate))
        self.final_hold_steps = max(1, int(self.final_hold_time * self.rate))

        # Explicit square vertices
        self.vertices = [
            [ self.half_side,  self.half_side, self.altitude],   # V1
            [ self.half_side, -self.half_side, self.altitude],   # V2
            [-self.half_side, -self.half_side, self.altitude],   # V3
            [-self.half_side,  self.half_side, self.altitude],   # V4
        ]

        self.path = []
        self.path_vertex_indices = []

        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.vehicle_odometry = VehicleOdometry()

        self.armed = False
        self.hit_shape = False
        self.offboard_setpoint_counter = 0
        self.offboard_arr_counter = 0
        self.start_time = time.time()
        self.shape_timer = None
        self.has_landed = False

        # =========================
        # Rosbag writer
        # =========================
        self.writer = None
        self.bag_uri = self.make_bag_uri("square_trace_bag")
        self.setup_rosbag_writer()

        # =========================
        # Build shape path
        # =========================
        self.init_square_path()

        # Main timer
        self.timer = self.create_timer(0.1, self.timer_callback)

    # -------------------------------------------------------------------------
    # Rosbag helpers
    # -------------------------------------------------------------------------
    def make_bag_uri(self, prefix: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"

    def setup_rosbag_writer(self) -> None:
        """Create rosbag writer and register all topics we want to record."""
        self.writer = rosbag2_py.SequentialWriter()

        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_uri,
            storage_id='sqlite3'
        )
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        self.register_bag_topic('/fmu/out/vehicle_odometry', 'px4_msgs/msg/VehicleOdometry')
        self.register_bag_topic('/fmu/out/vehicle_local_position', 'px4_msgs/msg/VehicleLocalPosition')
        self.register_bag_topic('/offboard_debug/reference', 'px4_msgs/msg/TrajectorySetpoint')
        self.register_bag_topic('/offboard_debug/active_vertex', 'std_msgs/msg/Int32')

        self.get_logger().info(f"Recording rosbag to: {self.bag_uri}")

    def register_bag_topic(self, name: str, type_str: str) -> None:
        """
        Register topic for rosbag writing.
        Compatible with older and newer ROS 2 TopicMetadata constructors.
        """
        try:
            topic_info = rosbag2_py.TopicMetadata(
                id=0,
                name=name,
                type=type_str,
                serialization_format='cdr'
            )
        except TypeError:
            topic_info = rosbag2_py.TopicMetadata(
                name=name,
                type=type_str,
                serialization_format='cdr'
            )

        self.writer.create_topic(topic_info)

    def bag_write(self, topic_name: str, msg) -> None:
        """Serialize and write a message to the bag."""
        if self.writer is None:
            return

        try:
            self.writer.write(
                topic_name,
                serialize_message(msg),
                self.get_clock().now().nanoseconds
            )
        except Exception as exc:
            self.get_logger().error(f"Failed bag write on {topic_name}: {exc}")

    # -------------------------------------------------------------------------
    # Path generation
    # -------------------------------------------------------------------------
    def init_square_path(self) -> None:
        """Generate interpolated setpoints for a square trajectory."""
        self.path = []
        self.path_vertex_indices = []

        self.get_logger().info("Square vertices:")
        for i, v in enumerate(self.vertices, start=1):
            self.get_logger().info(f"V{i}: x={v[0]:.3f}, y={v[1]:.3f}, z={v[2]:.3f}")

        # Traverse V1 -> V2 -> V3 -> V4 -> V1
        for i in range(len(self.vertices)):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % len(self.vertices)]

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            yaw = math.atan2(dy, dx)

            # Hold at corner so shape is visually obvious
            for _ in range(self.corner_hold_steps):
                msg = TrajectorySetpoint()
                msg.position = [start[0], start[1], start[2]]
                msg.velocity = [0.0, 0.0, 0.0]
                msg.acceleration = [0.0, 0.0, 0.0]
                msg.yaw = yaw
                msg.yawspeed = 0.0
                self.path.append(msg)
                self.path_vertex_indices.append(i)

            # Linearly interpolate along the current edge
            vx = dx / self.edge_traverse_time
            vy = dy / self.edge_traverse_time

            for step in range(1, self.edge_steps + 1):
                alpha = step / self.edge_steps

                x = start[0] + alpha * dx
                y = start[1] + alpha * dy
                z = self.altitude

                msg = TrajectorySetpoint()
                msg.position = [x, y, z]
                msg.velocity = [vx, vy, 0.0]
                msg.acceleration = [0.0, 0.0, 0.0]
                msg.yaw = yaw
                msg.yawspeed = 0.0
                self.path.append(msg)
                self.path_vertex_indices.append(i)

    # -------------------------------------------------------------------------
    # Main callbacks
    # -------------------------------------------------------------------------
    def timer_callback(self) -> None:
        """Main offboard timer."""
        self.publish_offboard_control_heartbeat_signal()

        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.armed = True

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        if (self.start_time + self.takeoff_hold_time) > time.time():
            self.publish_reference_setpoint(0.0, 0.0, self.altitude, 0.0, active_vertex=-1)
        else:
            if not self.hit_shape:
                self.get_logger().info("Tracing square now")
                self.shape_timer = self.create_timer(1.0 / self.rate, self.offboard_move_callback)
                self.hit_shape = True

    def offboard_move_callback(self) -> None:
        """Publish square path points one-by-one."""
        if self.offboard_arr_counter < len(self.path):
            msg = self.path[self.offboard_arr_counter]
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

            self.trajectory_setpoint_publisher.publish(msg)
            self.reference_setpoint_publisher.publish(msg)
            self.bag_write('/offboard_debug/reference', msg)

            vertex_msg = Int32()
            vertex_msg.data = self.path_vertex_indices[self.offboard_arr_counter]
            self.active_vertex_publisher.publish(vertex_msg)
            self.bag_write('/offboard_debug/active_vertex', vertex_msg)

        else:
            self.publish_reference_setpoint(0.0, 0.0, self.altitude, 0.0, active_vertex=-1)

        if self.offboard_arr_counter == len(self.path) + self.final_hold_steps:
            if self.shape_timer is not None:
                self.shape_timer.cancel()
            self.land()

        self.offboard_arr_counter += 1

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition) -> None:
        self.vehicle_local_position = msg
        self.bag_write('/fmu/out/vehicle_local_position', msg)

    def vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        self.vehicle_odometry = msg
        self.bag_write('/fmu/out/vehicle_odometry', msg)

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    # -------------------------------------------------------------------------
    # PX4 command helpers
    # -------------------------------------------------------------------------
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
        if self.has_landed:
            return

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")
        self.has_landed = True

    def publish_reference_setpoint(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        active_vertex: int
    ) -> None:
        """
        Publish a position setpoint to PX4 and also mirror it to a debug topic + rosbag.
        """
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.acceleration = [0.0, 0.0, 0.0]
        msg.yaw = yaw
        msg.yawspeed = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.trajectory_setpoint_publisher.publish(msg)
        self.reference_setpoint_publisher.publish(msg)
        self.bag_write('/offboard_debug/reference', msg)

        vertex_msg = Int32()
        vertex_msg.data = active_vertex
        self.active_vertex_publisher.publish(vertex_msg)
        self.bag_write('/offboard_debug/active_vertex', vertex_msg)

    def publish_offboard_control_heartbeat_signal(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

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
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def destroy_node(self):
        """
        Explicitly release bag writer before shutting down the node so metadata is flushed.
        """
        self.get_logger().info("Closing square rosbag writer")
        self.writer = None
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OffboardSquareBaggedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
