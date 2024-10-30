import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
from bb_msgs.msg import Ping # check if this is the correct path
from bb_perception_msgs.msg import (
    DetectedObject3D,
    DetectedObject3DArray,
    DetectorSource,
    ObjectHypothesis,
)
from collections import deque
from std_srvs.srv import SetBool 

""" pinger msg
int32 doa
int32 elevation
int32 frequency
float32 confidence
"""
"""DetectedObject3D msg
ObjectHypothesis hypothesis 
uint9 color 
string id 
"""

""" DectectedObject3D Array
std_msgs/Header header
string name
# The source of the detection. CAMERA/STEREO/SYNTHETIC_APERTURE
DetectorSource source
geometry_msgs/Pose sensor_pose
DetectedObject3D[] objects
"""

class GateStatusNode(Node):
    def __init__(self):
        super().__init__('pinger_gate_node')
        
        # expose some var as param 
        self.declare_parameter("pinger_data", "/sensors/ping")
        pinger_data = self.get_parameter("pinger_data").get_parameter_value().string_value
        
        # taking some doa to average for best result 
        self.declare_parameter("doa_cluster_size", 5)
        self.doa_cluster_size = self.get_parameter("doa_cluster_size").get_parameter_value().integer_value

        # Subscribers
        self.create_subscription(
            Odometry,
            "/asv4/nav/world",
            self.odom_callback,
            10,
        )
        self.create_subscription(
            DetectedObject3DArray, 
            "/asv4/vision/gate_detections", 
            self.gate_callback,
            10,
        )
        self.create_subscription(
            Ping, 
            pinger_data,
            self.pinger_callback, 
            10,
        )

        # Publishers
        self.pub_gate_ = self.create_publisher(String, 'pinger_gate', 10)
        self.pub_status_ = self.create_publisher(Bool, 'pinger_gate_status', 10)
        
        self.service = self.create_service(SetBool, 'activate_gate_node', self.service_callback)

        # Variables to store the latest data
        self.gate_data = None
        self.vehicle_position = None
        self.pinger_data = None 
        self.doa_values = deque(maxlen=self.doa_cluster_size) # store last N doa values 

        # Timer to periodically check data and publish status
        self.timer = self.create_timer(0.5, self.process_data)
        self.timer.cancel()  # Start with the timer off
        self.get_logger().info(f"Pinger Gate Node Initialised, waiting for activation")

    def gate_callback(self, msg):
        self.gate_detection = msg
        self.get_logger().info(f'Received gate data: {msg}')

    def odom_callback(self, msg):
        self.vehicle_position = msg.pose.pose.position
        self.get_logger().info(f'Received odometry data.')

    def pinger_callback(self, msg):
        """ taking only the doa, elevation data and others not accurate
        """
        self.pinger_data = msg
        self.doa_values.append(msg.doa)
        self.get_logger().info(f'Received pinger data: DOA={msg.doa}')
    
    def compute_average_doa(self):
        """Compute the average of the stored DOA values."""
        if len(self.doa_values) < self.doa_cluster_size :
            self.get_logger().info(f"doa cluster not filled, average not accurate, aborting")
            return 0  # Default value if no DOA values are available
        return sum(self.doa_values) / len(self.doa_values)

    def process_data(self):
        """Process the latest data and publish gate status."""
        if not self.is_active:
            return  # Do nothing if the node is not activated
        
        if not self.gate_detection or not self.vehicle_position or not self.pinger_data:
            self.get_logger().warn('Waiting for all sensor data...')
            return
        self.get_logger().info(f"All sensor data received. fake process data node")

        # Determine gate position using gpinger & gate data
        pinger_gate = self.determine_pinger_gate(self.gate_detection, self.vehicle_position, self.pinger_data)
        confidence_ok = self.check_pinger_gate_status(self.pinger_data)

        # Publish the gate position as a string
        pinger_gate_msg = String()
        pinger_gate_msg.data = pinger_gate
        self.pub_gate_.publish(pinger_gate_msg)

        # Publish the confidence status as a boolean
        confidence_msg = Bool()
        confidence_msg.data = confidence_ok
        self.pub_status_.publish(confidence_msg)

        # self.get_logger().info(f'Published: {pinger_gate}, Confidence OK: {confidence_ok}')

    def determine_pinger_gate(self, gate_detection, vehicle_position, pinger_data):
        """ Assumes asv is in front of the gate & stationary """
        """TODO : Check asv in front of the gate"""
        avg_doa = self.compute_average_doa()
        if 340 < avg_doa < 25 : return "gate_middle"
        elif 25 <= avg_doa <= 90 : return "gate_right"
        elif 270 <= avg_doa <= 240 : return "gate_left"
        self.get_logger().info(f"Invalid DOA, gate not determined")
        return 0

    def check_pinger_gate_status(self, pinger_data):
        """TODO"""
        return pinger_data.confidence > 0.8  # Threshold for confidence
    
    def service_callback(self, request, response):
        """Service callback to activate or deactivate the node."""
        self.is_active = request.data 

        if self.is_active:
            self.timer.reset() 
            response.message = "Gate status node activated."
        else:
            self.timer.cancel() 
            response.message = "Gate status node deactivated."

        response.success = True
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GateStatusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()