#include <rclcpp/rclcpp.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <bb_perception_msgs/msg/detected_object3_d.hpp>
#include <bb_perception_msgs/msg/detected_object3_d_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <unordered_map>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

class DetectedObject3DArrayVisNode : public rclcpp::Node {
public:
    DetectedObject3DArrayVisNode(
        const rclcpp::NodeOptions & options
    )
    : Node("detected_object_3d_visualization_node", options) {
        // Declare parameters
        this->declare_parameter<std::vector<std::string>>("input_detections_topics", {"/asv4/vision/lidar_small_objects/dets_3d"});
        this->declare_parameter<std::string>("output_markers_topic", "/asv4/vision/detections_2d/marker");
        this->declare_parameter<std::string>("objects_config", "robotx.yaml");
        this->declare_parameter<bool>("publish_tf", false);
        this->declare_parameter<bool>("publish_tf_unique", false);

        // Get parameters
        this->input_detections_topics_ = this->get_parameter("input_detections_topics").as_string_array();
        this->output_markers_topic_ = this->get_parameter("output_markers_topic").as_string();
        this->publish_tf_ = this->get_parameter("publish_tf").as_bool();
        this->publish_tf_unique_ = this->get_parameter("publish_tf_unique").as_bool();

        // Initialize publisher
        publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(output_markers_topic_, 10);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // Subscribe to topics for camera info and 3D detections
        for (const auto& topic : input_detections_topics_) {
            detection_subscribers_.push_back(this->create_subscription<bb_perception_msgs::msg::DetectedObject3DArray>(
                topic, 10, std::bind(&DetectedObject3DArrayVisNode::callback, this, std::placeholders::_1)
            ));
        }

        // Load config file (YAML format)
        load_config();
    }

private:
    // Load object configuration (e.g., from robotx.yaml)
    void load_config() {
        std::string config_path = ament_index_cpp::get_package_share_directory("ml_detector") + "/configs/objects/" + this->get_parameter("objects_config").as_string();
        try {
            YAML::Node config = YAML::LoadFile(config_path);
            if (config.IsNull()) {
                RCLCPP_ERROR(this->get_logger(), "Yaml config missing!");
                return;
            }
            for (const auto& obj : config["objects"]) {
                int label = obj["label"].as<int>();
                std::string name = obj["name"].as<std::string>();
                id_to_name_[label] = name;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load object config: %s", e.what());
        }
    }

    // Handle detection message and create markers
    void callback(const bb_perception_msgs::msg::DetectedObject3DArray::SharedPtr detection_msg) {
        auto markers = visualization_msgs::msg::MarkerArray();
        int i = 0;
        for (const auto& detection : detection_msg->objects) {
            std::string class_name = get_class_name(detection.hypothesis.class_id);
            auto marker = create_marker(detection, class_name, i++);
            markers.markers.push_back(marker);

            // Optionally publish TF
            if (publish_tf_) {
                publish_tf(detection, class_name, detection_msg->header, detection_msg->sensor_pose);
            }
        }
        publisher_->publish(markers);
    }

    visualization_msgs::msg::Marker create_marker(
        const bb_perception_msgs::msg::DetectedObject3D& detection,
        const std::string& class_name, int id) {

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = detection.hypothesis.kinematics.header.frame_id;
        marker.header.stamp = detection.hypothesis.kinematics.header.stamp;
        marker.ns = class_name;
        marker.id = id;

        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        
        if (class_name.find("cylinder") != std::string::npos) {
            marker.type = visualization_msgs::msg::Marker::CYLINDER;
        } else if (class_name.find("sphere") != std::string::npos) {
            marker.type = visualization_msgs::msg::Marker::SPHERE;
        }

        marker.pose = detection.hypothesis.kinematics.pose_with_covariance.pose;

        // Adjust z position
        marker.pose.position.z += detection.hypothesis.shape.dimensions.z / 2.0;

        // Set scale
        marker.scale = detection.hypothesis.shape.dimensions;
        marker.scale.x = std::max(0.5, marker.scale.x);
        marker.scale.y = std::max(0.5, marker.scale.y);
        marker.scale.z = std::max(0.5, marker.scale.z);

        int tid = detection.hypothesis.track_id;

        // Set color based on detection mode
        if (detection.hypothesis.mode == bb_perception_msgs::msg::ObjectHypothesis::MODE_DETECTED) {
            marker.color = get_color(id);
            marker.color.a = 0.2;
        } else {
            marker.color = get_color(tid);
        }

        // Handle color by class_name
        if (class_name.find("red") != std::string::npos) {
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        } else if (class_name.find("green") != std::string::npos) {
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        } else if (class_name.find("blue") != std::string::npos) {
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;
        } else if (class_name.find("black") != std::string::npos) {
            marker.color.r = marker.color.g = marker.color.b = 0.0;
        } else if (class_name.find("light_tower") != std::string::npos) {
            marker.color.r = marker.color.g = marker.color.b = 0.2;
        }

        marker.lifetime = rclcpp::Duration(1, 0);
        return marker;
    }

    visualization_msgs::msg::Marker create_text_marker(
        const visualization_msgs::msg::Marker& object_marker,
        const std::string& class_name, int id, int track_id) {

        visualization_msgs::msg::Marker text_marker;
        text_marker.header = object_marker.header;
        text_marker.ns = class_name + "_text";
        text_marker.id = id + 1000;  // Unique ID offset for text
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;

        // Text content
        if (track_id) {
            text_marker.text = class_name + ": " + std::to_string(track_id);
        } else {
            text_marker.text = class_name;
        }

        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.pose = object_marker.pose;
        text_marker.pose.position.z += object_marker.scale.z + 0.5;  // Place text above object

        // Set scale and color
        text_marker.scale.z = 1.0;
        text_marker.color = object_marker.color;
        text_marker.color.a = 1.0;  // Fully opaque text

        text_marker.lifetime = object_marker.lifetime;

        return text_marker;
    }

    void broadcast_transform(
        const bb_perception_msgs::msg::DetectedObject3D& detection,
        const std::string& class_name, int id) {

        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = detection.hypothesis.kinematics.header.stamp;
        transform.header.frame_id = detection.hypothesis.kinematics.header.frame_id;
        transform.child_frame_id = class_name + "_" + std::to_string(id);

        transform.transform.translation.x = detection.hypothesis.kinematics.pose_with_covariance.pose.position.x;
        transform.transform.translation.y = detection.hypothesis.kinematics.pose_with_covariance.pose.position.y;
        transform.transform.translation.z = detection.hypothesis.kinematics.pose_with_covariance.pose.position.z;
        transform.transform.rotation = detection.hypothesis.kinematics.pose_with_covariance.pose.orientation;

        tf_broadcaster_->sendTransform(transform);
    }


    // Generate color based on object ID
    std_msgs::msg::ColorRGBA get_color(int id) {
        std_msgs::msg::ColorRGBA color;
        color.r = static_cast<float>((id >> 16) % 256) / 255.0;
        color.g = static_cast<float>((id >> 8) % 256) / 255.0;
        color.b = static_cast<float>(id % 256) / 255.0;
        color.a = 1.0;
        return color;
    }

    // Publish TF for detected object
    void publish_tf(const bb_perception_msgs::msg::DetectedObject3D& detection, const std::string& class_name, const std_msgs::msg::Header& header, const geometry_msgs::msg::Pose& sensor_pose) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header = header;
        transform.child_frame_id = publish_tf_unique_ ? class_name : class_name + "_" + std::to_string(detection.hypothesis.class_id);
        transform.transform.translation.x = sensor_pose.position.x;
        transform.transform.translation.y = sensor_pose.position.y;
        transform.transform.translation.z = sensor_pose.position.z;
        transform.transform.rotation = sensor_pose.orientation;
        tf_broadcaster_->sendTransform(transform);
    }

    // Get class name from object ID
    std::string get_class_name(int class_id) {
        auto it = id_to_name_.find(class_id);
        return it != id_to_name_.end() ? it->second : "unknown";
    }

    std::unordered_map<int, std::string> id_to_name_;
    std::vector<std::string> input_detections_topics_;
    std::string output_markers_topic_;
    bool publish_tf_;
    bool publish_tf_unique_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::vector<rclcpp::Subscription<bb_perception_msgs::msg::DetectedObject3DArray>::SharedPtr> detection_subscribers_;
};

// int main(int argc, char** argv) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<DetectedObject3DArrayVisNode>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
RCLCPP_COMPONENTS_REGISTER_NODE(
    DetectedObject3DArrayVisNode
)