import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge
import threading

# Import the existing manager
from src.navigation_manager_node import NavigationManager

class ROS2NavigationNode(Node):
    def __init__(self):
        super().__init__('multimodal_navigation_node')
        self.bridge = CvBridge()

        # ROS Parameters
        self.declare_parameter('mock_camera', False)
        self.declare_parameter('mock_pose', False)

        mock_camera = self.get_parameter('mock_camera').get_parameter_value().bool_value
        mock_pose = self.get_parameter('mock_pose').get_parameter_value().bool_value

        self.get_logger().info(f"Initializing MINS Navigation Manager (Mock Camera: {mock_camera}, Mock Pose: {mock_pose})")

        self.manager = NavigationManager(mock_camera=mock_camera, mock_pose=mock_pose)

        # Publishers
        self.depth_pub = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)

        # Subscribers
        if not mock_camera:
            self.image_sub = self.create_subscription(
                Image,
                '/camera/color/image_raw',
                self.image_callback,
                10
            )

        if not mock_pose:
            self.pose_sub = self.create_subscription(
                String,
                '/rtabmap/localization_pose',
                self.pose_callback,
                10
            )

        # Start manager in a separate thread so it doesn't block ROS spin
        self.manager_thread = threading.Thread(target=self.manager.start, daemon=True)
        self.manager_thread.start()

        # Depth publisher loop
        self.depth_timer = self.create_timer(0.5, self.publish_depth_callback)
        self.last_depth_map = None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.manager.latest_frame = cv_image

            # Calculate and store depth for publishing
            if self.manager.latest_frame is not None:
                self.last_depth_map = self.manager.depth_estimator.estimate(cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def publish_depth_callback(self):
        if self.last_depth_map is not None:
            try:
                # Convert normalized depth [0,1] to standard metric float (meters approximation)
                # To match standard ROS 32FC1 depth topics. We'll simulate 0.5m to 5.0m
                metric_depth = (1.0 - self.last_depth_map) * 5.0
                metric_depth = metric_depth.astype('float32')
                msg = self.bridge.cv2_to_imgmsg(metric_depth, "32FC1")
                self.depth_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish depth: {e}")

    def pose_callback(self, msg):
        self.manager.current_node = msg.data

def main(args=None):
    rclpy.init(args=args)
    node = ROS2NavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.manager.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
