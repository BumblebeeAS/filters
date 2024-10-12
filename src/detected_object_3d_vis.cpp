#include <rclcpp/rclcpp.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <bb_perception_msgs/msg/detected_object3_d.hpp>
#include <bb_perception_msgs/msg/detected_object3_d_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <unordered_map>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

class DetectedObject3DArrayVisNode : public rclcpp::Node
{
public:
  DetectedObject3DArrayVisNode(const rclcpp::NodeOptions& options) :
    Node("detected_object_3d_visualization_node", options)
  {
    // Declare and get parameters
    this->declare_parameter<std::vector<std::string>>(
        "input_detections_topics", {"/asv4/vision/lidar_small_objects/dets_3d"});
    this->declare_parameter<std::string>(
        "output_markers_topic", "/asv4/vision/detections_2d/marker");
    this->declare_parameter<std::string>("objects_config", "robotx.yaml");
    this->declare_parameter<bool>("publish_tf", false);
    this->declare_parameter<bool>("publish_tf_unique", false);

    this->input_detections_topics_ =
        this->get_parameter("input_detections_topics").as_string_array();
    this->output_markers_topic_ = this->get_parameter("output_markers_topic").as_string();
    this->publish_tf_ = this->get_parameter("publish_tf").as_bool();
    this->publish_tf_unique_ = this->get_parameter("publish_tf_unique").as_bool();

    // Initialize publisher and broadcaster
    publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        output_markers_topic_, rclcpp::QoS(10).reliable());
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // Subscribe to topics
    for (const auto& topic : input_detections_topics_)
    {
      detection_subscribers_.push_back(
          this->create_subscription<bb_perception_msgs::msg::DetectedObject3DArray>(
              topic, 10,
              std::bind(
                  &DetectedObject3DArrayVisNode::callback, this, std::placeholders::_1)));
    }

    // Load configuration file
    load_config();
  }

private:
  // Load object configuration from YAML
  void load_config()
  {
    std::string config_path =
        ament_index_cpp::get_package_share_directory("ml_detector") +
        "/configs/objects/" + this->get_parameter("objects_config").as_string();
    try
    {
      YAML::Node config = YAML::LoadFile(config_path);
      if (!config)
      {
        RCLCPP_ERROR(this->get_logger(), "Yaml config missing or incorrect format!");
        return;
      }
      for (const auto& obj : config["objects"])
      {
        int label = obj["label"].as<int>();
        std::string name = obj["name"].as<std::string>();
        id_to_name_[label] = name;
      }
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to load object config: %s", e.what());
    }
  }

  // Callback for detections
  void
  callback(const bb_perception_msgs::msg::DetectedObject3DArray::SharedPtr detection_msg)
  {
    auto markers = visualization_msgs::msg::MarkerArray();
    int i = 0;
    for (const auto& detection : detection_msg->objects)
    {
      std::string class_name = get_class_name(detection.hypothesis.class_id);
      auto marker = create_marker(detection, class_name, i++);
      markers.markers.push_back(marker);
      auto text_marker = create_text_marker(detection, class_name, i++);
      markers.markers.push_back(text_marker);

      // Optionally publish TF
      if (publish_tf_)
      {
        publish_tf(
            detection, class_name, detection_msg->header, detection_msg->sensor_pose);
      }
    }
    publisher_->publish(markers);
  }

  // Create a marker for visualization
  visualization_msgs::msg::Marker create_marker(
      const bb_perception_msgs::msg::DetectedObject3D& detection,
      const std::string& class_name, int id)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = detection.hypothesis.kinematics.header.frame_id;
    marker.header.stamp = detection.hypothesis.kinematics.header.stamp;
    marker.ns = class_name;
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::CUBE;

    // Adjust type for other shapes
    if (class_name.find("cylinder") != std::string::npos)
    {
      marker.type = visualization_msgs::msg::Marker::CYLINDER;
    }
    else if (class_name.find("sphere") != std::string::npos)
    {
      marker.type = visualization_msgs::msg::Marker::SPHERE;
    }

    marker.pose = detection.hypothesis.kinematics.pose_with_covariance.pose;
    marker.pose.position.z += detection.hypothesis.shape.dimensions.z / 2.0;

    // Set marker dimensions and color
    marker.scale = detection.hypothesis.shape.dimensions;
    marker.color = get_color(detection.hypothesis.track_id);

    marker.lifetime = rclcpp::Duration(1, 0);
    return marker;
  }

  visualization_msgs::msg::Marker create_text_marker(
      const bb_perception_msgs::msg::DetectedObject3D& detection,
      const std::string& class_name, int id)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = detection.hypothesis.kinematics.header.frame_id;
    marker.header.stamp = detection.hypothesis.kinematics.header.stamp;
    marker.ns = class_name + "/text";
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.text = class_name;
    if (detection.hypothesis.track_id > 0)
    {
      marker.text += "(" + std::to_string(detection.hypothesis.track_id) + ")";
    }
    marker.pose = detection.hypothesis.kinematics.pose_with_covariance.pose;
    marker.pose.position.z += detection.hypothesis.shape.dimensions.z + 0.5;
    marker.scale.z = 0.2;
    marker.color = get_color(detection.hypothesis.track_id);
    marker.lifetime = rclcpp::Duration(1, 0);
    return marker;
  }

  // Get class name from object ID
  std::string get_class_name(int class_id)
  {
    auto it = id_to_name_.find(class_id);
    return it != id_to_name_.end() ? it->second : "unknown";
  }

  // Publish TF for detected object
  void publish_tf(
      const bb_perception_msgs::msg::DetectedObject3D& detection,
      const std::string& class_name, const std_msgs::msg::Header& header,
      const geometry_msgs::msg::Pose& sensor_pose)
  {
    geometry_msgs::msg::TransformStamped transform;
    transform.header = header;
    transform.child_frame_id =
        publish_tf_unique_ ?
            class_name :
            class_name + "_" + std::to_string(detection.hypothesis.class_id);
    transform.transform.translation.x = sensor_pose.position.x;
    transform.transform.translation.y = sensor_pose.position.y;
    transform.transform.translation.z = sensor_pose.position.z;
    transform.transform.rotation = sensor_pose.orientation;
    transform.header.stamp = rclcpp::Clock().now();
    transform.header.frame_id = header.frame_id;
    tf_broadcaster_->sendTransform(transform);
  }
  std_msgs::msg::ColorRGBA hsl_to_rgb(float h, float s, float l)
  {
    std_msgs::msg::ColorRGBA color;

    auto hue2rgb = [](float p, float q, float t) {
      if (t < 0.0f)
        t += 1.0f;
      if (t > 1.0f)
        t -= 1.0f;
      if (t < 1.0f / 6.0f)
        return p + (q - p) * 6.0f * t;
      if (t < 1.0f / 2.0f)
        return q;
      if (t < 2.0f / 3.0f)
        return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
      return p;
    };

    float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
    float p = 2.0f * l - q;

    color.r = hue2rgb(p, q, fmod(h + 1.0f / 3.0f, 1.0f));
    color.g = hue2rgb(p, q, fmod(h, 1.0f));
    color.b = hue2rgb(p, q, fmod(h - 1.0f / 3.0f, 1.0f));
    color.a = 1.0f;

    return color;
  }

  std_msgs::msg::ColorRGBA get_color(int id)
  {
    float hue = fmod(
        static_cast<float>(id) * 0.6180339887f,
        1.0f);                  // Golden ratio-based distribution
    float saturation = 0.65f;   // Fixed saturation for vibrancy
    float lightness = 0.55f;    // Fixed lightness for pleasant brightness

    return hsl_to_rgb(hue, saturation, lightness);
  }

  std::unordered_map<int, std::string> id_to_name_;
  std::vector<std::string> input_detections_topics_;
  std::string output_markers_topic_;
  bool publish_tf_;
  bool publish_tf_unique_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::vector<
      rclcpp::Subscription<bb_perception_msgs::msg::DetectedObject3DArray>::SharedPtr>
      detection_subscribers_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(DetectedObject3DArrayVisNode)
