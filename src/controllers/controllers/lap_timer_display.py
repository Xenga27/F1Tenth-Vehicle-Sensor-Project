import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LapTimerDisplay(Node):

    def __init__(self):
        super().__init__('lap_timer_display')

        self.subscription = self.create_subscription(
            String,
            '/lap_info',
            self.lap_callback,
            10
        )

        self.get_logger().info('Waiting for lap information...')

    def lap_callback(self, msg):
        try:
            data = json.loads(msg.data)

            lap = data['lap']
            target = data['target_laps']
            lap_times = data['lap_times']
            total_time = data['total_time']
            finished = data['finished']

            print('\033[2J\033[H', end='')

            print('=' * 38)
            print('          F1TENTH LAP TIMER')
            print('=' * 38)

            print(f'\nCurrent Lap: {lap} / {target}\n')

            print('LAP TIMES')
            print('-' * 38)

            for i, lap_time in enumerate(lap_times, 1):
                print(f'Lap {i:2}: {lap_time:8.2f} seconds')

            print('-' * 38)
            print(f'Total Time: {total_time:.2f} seconds')

            if finished:
                print('\n*** 10 LAPS COMPLETED! ***')
                print('*** CAR STOPPED ***')

            print('=' * 38, flush=True)

        except (KeyError, ValueError, TypeError) as error:
            self.get_logger().error(f'Invalid lap data: {error}')


def main(args=None):
    rclpy.init(args=args)

    node = LapTimerDisplay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
