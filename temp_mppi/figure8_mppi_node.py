import math
import time
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)


class OffboardFigure8MPPINode(Node):
    """
    Single-file ROS 2 node that extends the original PX4 figure-8 offboard example
    with a high-level MPPI tracker.

    Design philosophy:
    - PX4 still performs the low-level position/velocity/attitude control.
    - This node acts as a high-level reference generator.
    - The nominal reference is the same analytic figure-8 trajectory.
    - MPPI runs at every control step and corrects the next short-horizon reference
      based on the current measured vehicle state.
    """

    def __init__(self) -> None:
        super().__init__("offboard_figure8_mppi_node")

        self.get_logger().info("Offboard Figure-8 MPPI Node Alive!")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_profile
        )
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_profile
        )
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_profile
        )

        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            qos_profile,
        )
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_profile,
        )

        self.control_rate_hz = 20.0
        self.control_dt = 1.0 / self.control_rate_hz
        self.takeoff_hold_s = 10.0
        self.post_track_hover_s = 5.0

        self.figure8_radius = 1.0
        self.figure8_cycle_s = 8.0
        self.figure8_altitude = -1.5
        self.figure8_total_s = self.figure8_cycle_s

        self.horizon_steps = 25
        self.num_samples = 256
        self.temperature = 8.0

        self.jerk_noise_std = np.array([1.2, 1.2, 0.6], dtype=np.float64)

        self.max_speed = 2.5
        self.max_acceleration = 3.0
        self.max_jerk = 6.0

        self.w_position = 25.0
        self.w_velocity = 6.0
        self.w_acceleration = 1.5
        self.w_control = 0.08
        self.w_control_delta = 0.6
        self.w_terminal_position = 40.0
        self.w_terminal_velocity = 10.0
        self.w_terminal_acceleration = 2.0
        self.w_speed_limit = 30.0
        self.w_acceleration_limit = 20.0
        self.w_jerk_limit = 10.0

        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.have_local_position = False

        self.current_position = np.zeros(3, dtype=np.float64)
        self.current_velocity = np.zeros(3, dtype=np.float64)
        self.current_acceleration = np.zeros(3, dtype=np.float64)

        self.last_velocity_for_accel = None
        self.last_velocity_timestamp_s = None
        self.accel_lowpass_alpha = 0.35

        self.nominal_jerk_sequence = np.zeros(
            (self.horizon_steps, 3), dtype=np.float64
        )

        self.offboard_setpoint_counter = 0
        self.offboard_enabled = False
        self.armed = False
        self.landing_command_sent = False

        self.start_time_wall = time.time()
        self.figure8_start_wall = None

        self.final_hover_position, _, _, self.final_hover_yaw = self.figure8_reference_at_time(
            self.figure8_total_s
        )

        # Random generator for MPPI sampling
        self._rng = np.random.RandomState(0)

        # Cached constants for dynamics rollout
        self._dt = self.control_dt
        self._dt2 = self._dt * self._dt
        self._dt3 = self._dt2 * self._dt
        self._half_dt2 = 0.5 * self._dt2
        self._one_sixth_dt3 = (1.0 / 6.0) * self._dt3

        self.timer = self.create_timer(self.control_dt, self.timer_callback)

    # -------------------------------------------------------------------------
    # Reference trajectory generation
    # -------------------------------------------------------------------------
    def figure8_reference_at_time(
        self, tau: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        tau_clamped = min(max(tau, 0.0), self.figure8_total_s)

        r = self.figure8_radius
        dadt = (2.0 * math.pi) / self.figure8_cycle_s
        a = (-math.pi / 2.0) + tau_clamped * dadt

        c = math.cos(a)
        s = math.sin(a)
        cc = c * c
        ss = s * s

        sspo = ss + 1.0
        ssmo = ss - 1.0
        sspos = sspo * sspo
        c2a = math.cos(2.0 * a)
        c4a = math.cos(4.0 * a)
        c2am3 = c2a - 3.0
        c2am3_cubed = c2am3 * c2am3 * c2am3

        position = np.array(
            [
                -(r * c * s) / sspo,
                (r * c) / sspo,
                self.figure8_altitude,
            ],
            dtype=np.float64,
        )

        velocity = np.array(
            [
                dadt * r * (ss * ss + ss + ssmo * cc) / sspos,
                -dadt * r * s * (ss + 2.0 * cc + 1.0) / sspos,
                0.0,
            ],
            dtype=np.float64,
        )

        acceleration = np.array(
            [
                -dadt
                * dadt
                * 8.0
                * r
                * s
                * c
                * ((3.0 * c2a) + 7.0)
                / c2am3_cubed,
                dadt
                * dadt
                * r
                * c
                * ((44.0 * c2a) + c4a - 21.0)
                / c2am3_cubed,
                0.0,
            ],
            dtype=np.float64,
        )

        yaw = math.atan2(velocity[1], velocity[0])

        return position, velocity, acceleration, yaw

    def figure8_reference_batch(
        self, tau_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized figure-8 reference generation for the whole horizon.
        """
        tau = np.clip(tau_array, 0.0, self.figure8_total_s)

        r = self.figure8_radius
        dadt = (2.0 * np.pi) / self.figure8_cycle_s
        a = (-np.pi / 2.0) + tau * dadt

        c = np.cos(a)
        s = np.sin(a)
        cc = c * c
        ss = s * s

        sspo = ss + 1.0
        ssmo = ss - 1.0
        sspos = sspo * sspo
        c2a = np.cos(2.0 * a)
        c4a = np.cos(4.0 * a)
        c2am3 = c2a - 3.0
        c2am3_cubed = c2am3 * c2am3 * c2am3

        position = np.empty((tau.shape[0], 3), dtype=np.float64)
        velocity = np.empty((tau.shape[0], 3), dtype=np.float64)
        acceleration = np.empty((tau.shape[0], 3), dtype=np.float64)

        position[:, 0] = -(r * c * s) / sspo
        position[:, 1] = (r * c) / sspo
        position[:, 2] = self.figure8_altitude

        velocity[:, 0] = dadt * r * (ss * ss + ss + ssmo * cc) / sspos
        velocity[:, 1] = -dadt * r * s * (ss + 2.0 * cc + 1.0) / sspos
        velocity[:, 2] = 0.0

        acceleration[:, 0] = (
            -dadt * dadt * 8.0 * r * s * c * ((3.0 * c2a) + 7.0) / c2am3_cubed
        )
        acceleration[:, 1] = (
            dadt * dadt * r * c * ((44.0 * c2a) + c4a - 21.0) / c2am3_cubed
        )
        acceleration[:, 2] = 0.0

        yaw = np.arctan2(velocity[:, 1], velocity[:, 0])

        return position, velocity, acceleration, yaw

    def build_reference_horizon(
        self, elapsed_in_track: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tau = elapsed_in_track + (np.arange(self.horizon_steps, dtype=np.float64) + 1.0) * self.control_dt
        return self.figure8_reference_batch(tau)

    # -------------------------------------------------------------------------
    # State estimation callbacks
    # -------------------------------------------------------------------------
    def vehicle_local_position_callback(
        self, vehicle_local_position: VehicleLocalPosition
    ) -> None:
        self.vehicle_local_position = vehicle_local_position
        self.have_local_position = True

        self.current_position = np.array(
            [
                float(vehicle_local_position.x),
                float(vehicle_local_position.y),
                float(vehicle_local_position.z),
            ],
            dtype=np.float64,
        )

        current_velocity = np.array(
            [
                float(vehicle_local_position.vx),
                float(vehicle_local_position.vy),
                float(vehicle_local_position.vz),
            ],
            dtype=np.float64,
        )
        self.current_velocity = current_velocity

        now_s = self.get_clock().now().nanoseconds * 1e-9

        if (
            self.last_velocity_for_accel is not None
            and self.last_velocity_timestamp_s is not None
        ):
            dt = now_s - self.last_velocity_timestamp_s

            if dt > 1e-3:
                raw_accel = (current_velocity - self.last_velocity_for_accel) / dt
                self.current_acceleration = (
                    self.accel_lowpass_alpha * raw_accel
                    + (1.0 - self.accel_lowpass_alpha) * self.current_acceleration
                )

        self.last_velocity_for_accel = current_velocity.copy()
        self.last_velocity_timestamp_s = now_s

    def vehicle_status_callback(self, vehicle_status: VehicleStatus) -> None:
        self.vehicle_status = vehicle_status

    # -------------------------------------------------------------------------
    # Mission phase helpers
    # -------------------------------------------------------------------------
    def engage_offboard_mode(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )
        self.offboard_enabled = True
        self.get_logger().info("Switching to offboard mode")

    def arm(self) -> None:
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
        )
        self.armed = True
        self.get_logger().info("Arm command sent")

    def land(self) -> None:
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.landing_command_sent = True
        self.get_logger().info("Switching to land mode")

    # -------------------------------------------------------------------------
    # MPPI core
    # -------------------------------------------------------------------------
    def run_mppi(
        self,
        x0: np.ndarray,
        v0: np.ndarray,
        a0: np.ndarray,
        ref_p: np.ndarray,
        ref_v: np.ndarray,
        ref_a: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized-over-samples MPPI with reduced allocation overhead.
        """
        num_samples = self.num_samples
        horizon_steps = self.horizon_steps

        noise = self._rng.normal(
            loc=0.0,
            scale=self.jerk_noise_std,
            size=(num_samples, horizon_steps, 3),
        )
        sampled_jerk = np.clip(
            self.nominal_jerk_sequence[None, :, :] + noise,
            -self.max_jerk,
            self.max_jerk,
        )

        sample_p = np.empty((num_samples, 3), dtype=np.float64)
        sample_v = np.empty((num_samples, 3), dtype=np.float64)
        sample_a = np.empty((num_samples, 3), dtype=np.float64)

        sample_p[:] = x0
        sample_v[:] = v0
        sample_a[:] = a0

        sample_cost = np.zeros(num_samples, dtype=np.float64)
        previous_jerk = np.zeros((num_samples, 3), dtype=np.float64)

        dt = self._dt
        half_dt2 = self._half_dt2
        one_sixth_dt3 = self._one_sixth_dt3

        for i in range(horizon_steps):
            jerk_i = sampled_jerk[:, i, :]

            next_p = sample_p + sample_v * dt + sample_a * half_dt2 + jerk_i * one_sixth_dt3
            next_v = sample_v + sample_a * dt + jerk_i * half_dt2
            next_a = sample_a + jerk_i * dt

            dp = next_p - ref_p[i]
            dv = next_v - ref_v[i]
            da = next_a - ref_a[i]

            tracking_cost = (
                self.w_position * np.einsum("ij,ij->i", dp, dp)
                + self.w_velocity * np.einsum("ij,ij->i", dv, dv)
                + self.w_acceleration * np.einsum("ij,ij->i", da, da)
            )

            jerk_delta = jerk_i - previous_jerk
            smoothness_cost = (
                self.w_control * np.einsum("ij,ij->i", jerk_i, jerk_i)
                + self.w_control_delta * np.einsum("ij,ij->i", jerk_delta, jerk_delta)
            )

            speed_norm = np.sqrt(np.einsum("ij,ij->i", next_v, next_v))
            accel_norm = np.sqrt(np.einsum("ij,ij->i", next_a, next_a))
            jerk_norm = np.sqrt(np.einsum("ij,ij->i", jerk_i, jerk_i))

            speed_violation = np.maximum(0.0, speed_norm - self.max_speed)
            accel_violation = np.maximum(0.0, accel_norm - self.max_acceleration)
            jerk_violation = np.maximum(0.0, jerk_norm - self.max_jerk)

            limit_cost = (
                self.w_speed_limit * speed_violation * speed_violation
                + self.w_acceleration_limit * accel_violation * accel_violation
                + self.w_jerk_limit * jerk_violation * jerk_violation
            )

            sample_cost += tracking_cost + smoothness_cost + limit_cost

            previous_jerk[:] = jerk_i
            sample_p[:] = next_p
            sample_v[:] = next_v
            sample_a[:] = next_a

        dp_terminal = sample_p - ref_p[-1]
        dv_terminal = sample_v - ref_v[-1]
        da_terminal = sample_a - ref_a[-1]

        sample_cost += (
            self.w_terminal_position * np.einsum("ij,ij->i", dp_terminal, dp_terminal)
            + self.w_terminal_velocity * np.einsum("ij,ij->i", dv_terminal, dv_terminal)
            + self.w_terminal_acceleration * np.einsum("ij,ij->i", da_terminal, da_terminal)
        )

        shifted_cost = sample_cost - np.min(sample_cost)
        weights = np.exp(-shifted_cost / self.temperature)
        weights_sum = np.sum(weights)

        if weights_sum < 1e-12:
            weights.fill(1.0 / num_samples)
        else:
            weights /= weights_sum

        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        self.nominal_jerk_sequence += weighted_noise
        np.clip(
            self.nominal_jerk_sequence,
            -self.max_jerk,
            self.max_jerk,
            out=self.nominal_jerk_sequence,
        )

        optimal_jerk_0 = self.nominal_jerk_sequence[0]

        desired_position = x0 + v0 * dt + a0 * half_dt2 + optimal_jerk_0 * one_sixth_dt3
        desired_velocity = v0 + a0 * dt + optimal_jerk_0 * half_dt2
        desired_acceleration = a0 + optimal_jerk_0 * dt

        self.nominal_jerk_sequence[:-1] = self.nominal_jerk_sequence[1:]
        self.nominal_jerk_sequence[-1] = self.nominal_jerk_sequence[-2]

        return desired_position, desired_velocity, desired_acceleration

    # -------------------------------------------------------------------------
    # Publishing helpers
    # -------------------------------------------------------------------------
    def publish_offboard_control_heartbeat_signal(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_takeoff_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(yaw)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_tracking_setpoint(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        yaw: float,
        yawspeed: float,
    ) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(position[0]), float(position[1]), float(position[2])]
        msg.velocity = [float(velocity[0]), float(velocity[1]), float(velocity[2])]
        msg.acceleration = [
            float(acceleration[0]),
            float(acceleration[1]),
            float(acceleration[2]),
        ]
        msg.yaw = float(yaw)
        msg.yawspeed = float(yawspeed)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

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

    # -------------------------------------------------------------------------
    # Utility helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def yaw_from_velocity(velocity: np.ndarray, fallback_yaw: float) -> float:
        planar_speed = math.hypot(float(velocity[0]), float(velocity[1]))
        if planar_speed < 1e-3:
            return float(fallback_yaw)
        return math.atan2(float(velocity[1]), float(velocity[0]))

    @staticmethod
    def unwrap_yaw_rate(yaw_now: float, yaw_next: float, dt: float) -> float:
        dyaw = yaw_next - yaw_now
        while dyaw > math.pi:
            dyaw -= 2.0 * math.pi
        while dyaw < -math.pi:
            dyaw += 2.0 * math.pi
        return dyaw / dt

    # -------------------------------------------------------------------------
    # Main timer callback
    # -------------------------------------------------------------------------
    def timer_callback(self) -> None:
        self.publish_offboard_control_heartbeat_signal()

        if self.offboard_setpoint_counter == 10 and not self.offboard_enabled:
            self.engage_offboard_mode()

        if self.offboard_setpoint_counter == 10 and not self.armed:
            self.arm()

        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

        now_wall = time.time()
        mission_elapsed = now_wall - self.start_time_wall

        if mission_elapsed < self.takeoff_hold_s:
            self.publish_takeoff_setpoint(
                x=0.0,
                y=0.0,
                z=self.figure8_altitude,
                yaw=math.radians(45.0),
            )
            return

        if self.figure8_start_wall is None:
            self.figure8_start_wall = now_wall
            self.get_logger().info("Starting MPPI figure-8 tracking")

        track_elapsed = now_wall - self.figure8_start_wall

        if track_elapsed <= self.figure8_total_s:
            if not self.have_local_position:
                self.get_logger().warn(
                    "Waiting for VehicleLocalPosition before running MPPI"
                )
                self.publish_takeoff_setpoint(
                    x=0.0,
                    y=0.0,
                    z=self.figure8_altitude,
                    yaw=math.radians(45.0),
                )
                return

            ref_p, ref_v, ref_a, ref_yaw = self.build_reference_horizon(track_elapsed)

            desired_p, desired_v, desired_a = self.run_mppi(
                x0=self.current_position,
                v0=self.current_velocity,
                a0=self.current_acceleration,
                ref_p=ref_p,
                ref_v=ref_v,
                ref_a=ref_a,
            )

            desired_yaw = self.yaw_from_velocity(desired_v, ref_yaw[0])

            if self.horizon_steps >= 2:
                desired_yawspeed = self.unwrap_yaw_rate(
                    ref_yaw[0], ref_yaw[1], self.control_dt
                )
            else:
                desired_yawspeed = 0.0

            self.publish_tracking_setpoint(
                position=desired_p,
                velocity=desired_v,
                acceleration=desired_a,
                yaw=desired_yaw,
                yawspeed=desired_yawspeed,
            )
            return

        if track_elapsed <= self.figure8_total_s + self.post_track_hover_s:
            self.publish_takeoff_setpoint(
                x=float(self.final_hover_position[0]),
                y=float(self.final_hover_position[1]),
                z=float(self.final_hover_position[2]),
                yaw=float(self.final_hover_yaw),
            )
            return

        if not self.landing_command_sent:
            self.land()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OffboardFigure8MPPINode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
