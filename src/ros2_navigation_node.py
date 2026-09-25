import threading
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# Core Subsystems
from src.perception.depth_estimator import DepthEstimator
from src.perception.yolo_detector import YOLOWorldDetector
from src.navigation.semantic_graph import SemanticGraph
from src.navigation.path_planner import SemanticConstrainedPlanner
from src.reasoning.llm_route_explainer import LLMRouteExplainer
from src.reasoning.vlm_guidance_node import VLMGuidanceNode
from src.interaction.tts_output import TTSOutput


class NavigationROSNode(Node):
    """
    Unified ROS 2 Orchestrator Node for Multimodal Navigation System.
    Integrates Depth Anything V2, YOLO-World, OSMAG Path Planning,
    Grounded VLM Guidance, and Preemptive Audio Feedback.
    """
    def __init__(self):
        super().__init__('multimodal_navigation_node')
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('graph_config', 'campus_graph.json')
        self.declare_parameter('yolo_model', 'yolov8s-world.pt')
        graph_file = self.get_parameter('graph_config').get_parameter_value().string_value
        yolo_path = self.get_parameter('yolo_model').get_parameter_value().string_value

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.pose_sub = self.create_subscription(
            String,
            '/rtabmap/localization_pose',
            self.pose_callback,
            10
        )
        self.voice_goal_sub = self.create_subscription(
            String,
            '/interaction/voice_goal',
            self.voice_goal_callback,
            10
        )

        # Publishers
        self.depth_pub = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self.guidance_pub = self.create_publisher(String, '/interaction/guidance', 10)
        self.alert_pub = self.create_publisher(String, '/interaction/alert', 10)
        self.status_pub = self.create_publisher(String, '/navigation/status', 10)

        # State Variables
        self.current_node = None
        self.goal_node = None
        self.path_nodes = []
        self._obstacle_history = {}
        self.processing_lock = threading.Lock()

        # Initialize Core Subsystems
        self.get_logger().info("Initializing Navigation Subsystems...")
        try:
            self.graph = SemanticGraph(graph_file)
            self.planner = SemanticConstrainedPlanner(self.graph)
            self.explainer = LLMRouteExplainer(self.graph)
            self.vlm_guidance = VLMGuidanceNode()
            self.depth_estimator = DepthEstimator()
            self.yolo = YOLOWorldDetector(model_path=yolo_path)
            self.tts = TTSOutput()
            self.get_logger().info("All navigation subsystems initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize subsystems: {e}")

    def pose_callback(self, msg: String):
        """Updates current topological semantic node ID from RTAB-Map localization."""
        new_node = msg.data.strip()
        if self.current_node != new_node:
            self.current_node = new_node
            self.get_logger().info(f"Localized at topological node: {self.current_node}")

    def voice_goal_callback(self, msg: String):
        """Processes spoken goal commands and computes the initial route."""
        spoken = msg.data.strip()
        self.get_logger().info(f"Received goal query: '{spoken}'")

        target_id = None
        for n_id, node in self.graph.nodes.items():
            name = node.get("name", "").lower()
            if n_id.lower() in spoken.lower() or (name and name in spoken.lower()):
                target_id = n_id
                break

        if not target_id:
            self.tts.speak("Destination not recognized.")
            return

        self.goal_node = target_id
        if self.current_node:
            self._replan()
        else:
            self.tts.speak("Current location unknown. Waiting for localization.")

    def _replan(self):
        """Calculates topological path and generates high-level LLM guidance."""
        path_dict = self.planner.find_path(self.current_node, self.goal_node)
        self.path_nodes = path_dict.get("path", [])

        if not self.path_nodes:
            self.tts.speak("Cannot find a valid alternate route. Destination unreachable.")
            return

        summary = self.explainer.summarize_route(self.path_nodes)
        self.get_logger().info(f"Planned Route: {self.path_nodes}")
        self.get_logger().info(f"Route Explanation: {summary}")
        self.status_pub.publish(String(data=f"Active Path: {' -> '.join(self.path_nodes)}"))
        self.tts.speak(f"New route calculated. {summary}")

    def image_callback(self, msg: Image):
        """Perception pipeline: Depth -> YOLO-World -> Obstacle Safety -> Grounded VLM."""
        # Drop frame if previous heavy inference is still executing
        if not self.processing_lock.acquire(blocking=False):
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 1. Depth Estimation (Depth Anything V2)
            depth_map = self.depth_estimator.estimate(frame)

            # Publish synthetic depth map (32FC1 in meters) for RTAB-Map or monitoring
            try:
                metric_depth = depth_map.astype('float32')
                depth_msg = self.bridge.cv2_to_imgmsg(metric_depth, encoding="32FC1")
                depth_msg.header = msg.header
                self.depth_pub.publish(depth_msg)
            except Exception as e:
                self.get_logger().warn(f"Failed to publish depth frame: {e}")

            # 2. Open-Vocabulary Detection & 3D Proximity Fusion
            detections, changed, immediate_hazard = self.yolo.detect(
                frame, depth_map, self.depth_estimator
            )

            # 3. Collision Preemption & Dynamic Graph Updating
            current_detected_classes = set()
            if immediate_hazard:
                hazards = [d for d in detections if d["direction"] == "Center" and d["distance"] < 1.5]
                if hazards:
                    closest = min(hazards, key=lambda x: x["distance"])
                    alert_text = f"Caution: {closest['class']} {closest['distance']:.1f} meters directly ahead."

                    # Priority safety broadcast & voice interrupt
                    self.alert_pub.publish(String(data=alert_text))
                    self.tts.speak_alert(alert_text)

                    hazard_class = closest['class']
                    current_detected_classes.add(hazard_class)
                    self._obstacle_history[hazard_class] = self._obstacle_history.get(hazard_class, 0) + 1

                    # If an obstacle persists across 20 frames, mark next node out-of-service and reroute
                    if self._obstacle_history[hazard_class] >= 20 and self.path_nodes and self.current_node:
                        try:
                            idx = self.path_nodes.index(self.current_node)
                            if idx + 1 < len(self.path_nodes):
                                next_node = self.path_nodes[idx + 1]
                                self.get_logger().warn(
                                    f"Persistent obstacle '{hazard_class}' detected. "
                                    f"Marking node '{next_node}' out of service."
                                )
                                self.graph.mark_node_out_of_service(next_node)
                                self._replan()
                                self._obstacle_history[hazard_class] = 0
                        except ValueError:
                            pass

            # Decay tracking for cleared obstacles
            for cls in list(self._obstacle_history.keys()):
                if cls not in current_detected_classes:
                    self._obstacle_history[cls] = 0

            # 4. Perception-Grounded VLM Real-Time Guidance
            if self.path_nodes and self.current_node:
                try:
                    idx = self.path_nodes.index(self.current_node)
                    next_node = self.path_nodes[idx + 1] if idx + 1 < len(self.path_nodes) else self.current_node

                    curr_node_info = self.graph.get_node(self.current_node) or {}
                    next_node_info = self.graph.get_node(next_node) or {}

                    curr_name = curr_node_info.get("name", self.current_node)
                    next_name = next_node_info.get("name", next_node)

                    det_hash = f"{len(detections)}_" + "_".join(sorted([d['class'] for d in detections]))

                    if self.vlm_guidance.should_reprompt(self.current_node, next_node, det_hash):
                        guidance = self.vlm_guidance.generate_guidance(
                            frame, curr_name, next_name, detections, det_hash
                        )
                        if guidance:
                            self.guidance_pub.publish(String(data=guidance))
                            self.tts.speak(guidance)
                except ValueError:
                    pass

        except Exception as e:
            self.get_logger().error(f"Error in perception callback: {e}")
        finally:
            self.processing_lock.release()


def main(args=None):
    rclpy.init(args=args)
    node = NavigationROSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()