#!/usr/bin/env python3
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
from collections import Counter

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

"""TODO: stop node when no new acoustic data received"""

class GateStatusNode(Node):
    def __init__(self):
        super().__init__('pinger_gate_node')
        
        # expose some var as param 
        self.declare_parameter("pinger_data", "/sensors/ping")
        pinger_data = self.get_parameter("pinger_data").get_parameter_value().string_value
        
        # taking some doa to average for best result 
        self.declare_parameter("doa_cluster_size", 10)
        self.doa_cluster_size = self.get_parameter("doa_cluster_size").get_parameter_value().integer_value
        self.get_logger().info(f"DOA cluster size set to: {self.doa_cluster_size}")

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
    
        # assuming 5m in front and facing gate straight head 
        self.gate_left_threshold = (270,320) # red - white 
        self.gate_middle_threshold = (325,40) # white - white 
        self.gate_right_threshold = (45,75) # white - green
        # Publishers
        self.pub_gate_ = self.create_publisher(String, '/asv4/robotx/pinger_gate', 10)
        self.pub_status_ = self.create_publisher(Bool, '/asv4/robotx/pinger_gate_status', 10)
        
        self.service = self.create_service(SetBool, '/robotx/activate_pinger_gate_node', self.service_callback)

        # Variables to store the latest data
        self.gate_detection = None
        self.vehicle_position = None
        self.pinger_data = None 
        # self.doa_values = deque(maxlen=self.doa_cluster_size) # store last N doa values 
        self.batch = []
        
        #  self.timer = self.create_timer(0.5, self.process_data) # move to pinger callback
        self.is_active = False  # Start with the timer off
        self.get_logger().info(f"Pinger Gate Node Initialised, waiting for activation")

    def gate_callback(self, msg):
        self.gate_detection = msg
        # self.get_logger().info(f'Received gate data: {msg}')

    def odom_callback(self, msg):
        self.vehicle_position = msg.pose.pose.position
        # self.get_logger().info(f'Received odometry data.'throttle_duration_sec=2.0)
        
    def get_gate_from_doa(self,doa):
        if self.gate_middle_threshold[0] >= doa and doa <= self.gate_middle_threshold[1] : 
           return  "gate_middle"
        elif self.gate_right_threshold[0] <= doa <= self.gate_right_threshold[1] : 
            return "gate_right" 
        elif self.gate_left_threshold[0] <= doa <= self.gate_left_threshold[1] : 
            return "gate_left" 
        else:
            self.get_logger().info(f"Invalid doa")
            return

    def pinger_callback(self, msg):
        """ taking only the doa, elevation data and others not accurate
        """
         # self.get_logger().info(f'Received pinger data: DOA={msg.doa}')
  
        if not self.is_active:
            return 
    
        self.batch.append(self.get_gate_from_doa(msg.doa))
        if len(self.batch) == self.doa_cluster_size:
            gate = Counter(self.batch).most_common(1)[0][0]
            self.get_logger().info(f"Gate is: {gate}")
            
            gate_pub = String()
            if gate is None:
                return
            gate_pub.data = gate
            self.pub_gate_.publish(gate_pub)
            self.batch = []
            
     
    # def compute_average_doa(self):
    #     """Compute the average of the stored DOA values."""
    #     self.get_logger().info(f"DOA values: {self.doa_values}")
        
    #     if len(self.doa_values) < self.doa_cluster_size :
    #         return -1  # Default value if no DOA values are available
    #     return sum(self.doa_values) / len(self.doa_values)

    # def process_data(self):
    #     if not self.is_active:
    #         return 
            
    #     # pinger_gate = self.determine_pinger_gate(self.gate_detection, self.vehicle_position, self.pinger_data)
    #     pinger_gate = self.determine_pinger_gate()
    #     confidence_ok = self.check_pinger_gate_status(self.pinger_data)

    #     # Publish the gate position as a string
    #     pinger_gate_msg = String()
    #     pinger_gate_msg.data = pinger_gate
    #     self.timer = self.create_timer(1.0 , self.pub_gate_.publish(pinger_gate_msg))

    #     # Publish the confidence status as a boolean
    #     confidence_msg = Bool()
    #     confidence_msg.data = confidence_ok
    #     self.pub_status_.publish(confidence_msg)

        # self.get_logger().info(f'Published: {pinger_gate}, Confidence OK: {confidence_ok}')

    def check_pinger_gate_status(self, pinger_data):
        """TODO"""
        return pinger_data.confidence > 0.8  # Threshold for confidence
    
    def service_callback(self, request, response):
        """Service callback to activate or deactivate the node."""
        self.is_active = request.data  # Set the node state based on the request

        if self.is_active:
            response.message = "Gate status node activated."
            self.get_logger().info('Node activated.')
        else:
            response.message = "Gate status node deactivated."
            self.get_logger().info('Node deactivated.')

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
