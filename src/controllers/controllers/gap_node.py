import math
import time
import json

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

class ReactiveFollowGap(Node):

    def __init__(self):
        super().__init__('follow_gap_controller')
        # Lap counter: virtual start/finish gate
        self.target_laps = 10
        self.lap_count = 0
        # Lap timing
        self.start_time = None
        self.last_lap_time = None
        self.lap_times = []

        self.total_distance = 0.0
        self.last_lap_distance = 0.0
        self.min_lap_distance = 50.0

        self.prev_x = None
        self.prev_y = None
        self.prev_s = None

        self.start_x = None
        self.start_y = None
        self.start_yaw = None

        self.gate_armed = False
        self.arm_distance = 5.0
        self.gate_lateral_tolerance = 2.0
        self.gate_heading_tolerance = 0.6

        self.odom_received = False
        self.finished = False

        # Start/finish gate at x=0, y=0
        self.gate_y_tolerance = 2.0
        self.gate_heading_tolerance = 0.5

        # Vehicle settings
        self.max_speed = 15.28
        self.min_speed = 2.0
        self.max_steering = 0.4189

        # Follow-the-Gap settings
        self.safety_radius = 0.40
        self.emergency_distance = 0.25
        self.front_angle = 0.65

        self.last_steering = 0.0

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/ego_racecar/odom',
            self.odom_callback,
            10
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            '/drive',
            10
        )
        # Publish lap information for the separate timer terminal
        self.lap_info_pub = self.create_publisher(
            String,
            '/lap_info',
            10
        )

        self.get_logger().info(
            'Follow-the-Gap initialized. Target: 10 laps.'
        )

    def publish_drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)

    def odom_callback(self, msg):

        if self.finished:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Record the initial vehicle pose
        if self.start_x is None:
            self.start_x = x
            self.start_y = y
            self.start_yaw = yaw
            
            now = time.monotonic()
            self.start_time = now
            self.last_lap_time = now

            self.prev_x = x
            self.prev_y = y
            self.prev_s = 0.0

            self.odom_received = True

            self.get_logger().info(
                f'Virtual start gate set at ({x:.2f}, {y:.2f}), '
                f'yaw={yaw:.2f}'
            )
            return

        # Calculate traveled distance
        distance = math.hypot(
            x - self.prev_x,
            y - self.prev_y
        )

        self.total_distance += distance

        # Calculate distance from virtual start position
        dx = x - self.start_x
        dy = y - self.start_y

        distance_from_start = math.hypot(dx, dy)

        # Calculate heading difference
        yaw_difference = math.atan2(
            math.sin(yaw - self.start_yaw),
            math.cos(yaw - self.start_yaw)
        )

        # Arm gate after leaving the start area
        if distance_from_start > self.arm_distance:
            self.gate_armed = True

        # Detect return to the virtual start area
        crossed_gate = (
            self.gate_armed
            and distance_from_start <= self.gate_lateral_tolerance
            and abs(yaw_difference) <= self.gate_heading_tolerance
            and (
                self.total_distance - self.last_lap_distance
                >= self.min_lap_distance
            )
        )

        if crossed_gate:
            self.gate_armed = False
            self.lap_count += 1
            self.last_lap_distance = self.total_distance

            # Calculate lap and total time
            now = time.monotonic()
            lap_elapsed = now - self.last_lap_time
            total_elapsed = now - self.start_time

            self.last_lap_time = now
            self.lap_times.append(lap_elapsed)

            # Publish lap information
            lap_msg = String()
            lap_msg.data = json.dumps({
                'lap': self.lap_count,
                'target_laps': self.target_laps,
                'lap_time': lap_elapsed,
                'total_time': total_elapsed,
                'lap_times': self.lap_times,
                'finished': self.lap_count >= self.target_laps
            })
            self.lap_info_pub.publish(lap_msg)

            self.get_logger().info(
                f'Lap {self.lap_count}/{self.target_laps} | '
                f'Lap time: {lap_elapsed:.2f} s | '
                f'Total time: {total_elapsed:.2f} s'
            )

            if self.lap_count >= self.target_laps:
                self.finished = True
                self.publish_drive(0.0, 0.0)

                self.get_logger().info(
                    f'10 LAPS COMPLETED! CAR STOPPED! '
                    f'TOTAL TIME: {total_elapsed:.2f} seconds'
                )

        self.prev_x = x
        self.prev_y = y

    def lidar_callback(self, data):

        if self.finished:
            self.publish_drive(0.0, 0.0)
            return

        if not self.odom_received:
            self.publish_drive(0.0, 0.0)
            return

        ranges = np.asarray(data.ranges, dtype=np.float32)

        if len(ranges) == 0 or data.angle_increment == 0.0:
            self.publish_drive(0.0, 0.0)
            return

        # Replace invalid LiDAR readings
        ranges[~np.isfinite(ranges)] = data.range_max
        ranges[ranges <= 0.0] = data.range_max

        ranges = np.clip(
            ranges,
            data.range_min,
            data.range_max
        )

        # Smooth LiDAR readings
        window = 5
        padded = np.pad(ranges, (2, 2), mode='edge')

        ranges = np.convolve(
            padded,
            np.ones(window) / window,
            mode='valid'
        )

        # Consider only the forward field of view
        angles = (
            data.angle_min
            + np.arange(len(ranges)) * data.angle_increment
        )

        roi = np.where(
            np.abs(angles) <= self.front_angle
        )[0]

        if len(roi) == 0:
            self.publish_drive(0.0, 0.0)
            return

        front_ranges = ranges[roi].copy()
        front_angles = angles[roi]

        # Emergency stop
        closest_index = int(np.argmin(front_ranges))
        closest_distance = float(front_ranges[closest_index])

        if closest_distance < self.emergency_distance:
            self.publish_drive(0.0, 0.0)

            self.get_logger().warn(
                f'EMERGENCY STOP: {closest_distance:.2f} m'
            )
            return

        # Create safety bubble around the closest obstacle
        angle_increment = abs(data.angle_increment)

        bubble_angle = math.asin(
            min(
                1.0,
                self.safety_radius / max(closest_distance, 0.05)
            )
        )

        bubble_size = int(
            bubble_angle / angle_increment
        ) + 2

        start = max(0, closest_index - bubble_size)

        end = min(
            len(front_ranges),
            closest_index + bubble_size + 1
        )

        front_ranges[start:end] = 0.0

        # Find continuous free-space gaps
        free_mask = front_ranges > 0.05

        padded_mask = np.concatenate(
            ([False], free_mask, [False])
        )

        changes = np.diff(padded_mask.astype(int))

        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0] - 1

        if len(starts) == 0:
            self.publish_drive(0.0, 0.0)
            return

        # Select the largest gap
        lengths = ends - starts + 1
        best_gap = int(np.argmax(lengths))

        gap_start = starts[best_gap]
        gap_end = ends[best_gap]

        # Select the center of the largest gap
        if gap_end < gap_start:
            self.publish_drive(0.0, 0.0)
            return

        best_index = (gap_start + gap_end) // 2

        steering = float(front_angles[best_index])

        steering = float(np.clip(
            steering,
            -self.max_steering,
            self.max_steering
        ))

        # Smooth steering
        steering = (
            0.5 * self.last_steering
            + 0.5 * steering
        )

        self.last_steering = steering

        # Adaptive speed
        steering_factor = (
            abs(steering) / self.max_steering
        )

        speed = self.max_speed - (
            self.max_speed - self.min_speed
        ) * (min(steering_factor, 1.0) ** 0.6)

        # Slow down when an obstacle is close in front
        center = len(front_ranges) // 2

        front_window = front_ranges[
            max(0, center - 20):
            min(len(front_ranges), center + 21)
        ]

        front_distance = float(np.min(front_window))

        if front_distance < 1.5:
            speed = min(speed, 0.6)

        if front_distance < 0.8:
            speed = min(speed, 0.4)

        self.publish_drive(speed, steering)

        self.get_logger().info(
            f'Lap: {self.lap_count}/{self.target_laps} | '
            f'Speed: {speed:.2f} m/s | '
            f'Steering: {steering:.3f} rad | '
            f'Front: {front_distance:.2f} m'
        )


def main(args=None):

    rclpy.init(args=args)

    controller = ReactiveFollowGap()

    try:
        rclpy.spin(controller)

    except KeyboardInterrupt:
        pass
        

    finally:
        controller.publish_drive(0.0, 0.0)
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
