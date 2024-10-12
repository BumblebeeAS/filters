#include <rclcpp/rclcpp.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <bb_perception_msgs/msg/detected_object2_d.hpp>
#include <bb_perception_msgs/msg/detected_object2_d_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <unordered_map>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class DetectedObject2DArrayVisNode : public rclcpp::Node
{
public:
  DetectedObject2DArrayVisNode(const rclcpp::NodeOptions& options) :
    Node("detected_object_2d_visualization_node", options)
  {
    // Declare parameters
    this->declare_parameter<std::vector<std::string>>(
        "input_detections_topics", {"/asv4/vision/detections_2d"});
    this->declare_parameter<std::vector<std::string>>(
        "camera_info_topics", {
                                  "/asv4/left_cam/camera_info",
                                  "/asv4/right_cam/camera_info",
                                  "/asv4/front_cam/camera_info",
                              });
    this->declare_parameter<std::string>(
        "output_markers_topic", "/asv4/vision/detections_2d/marker");
    this->declare_parameter<std::string>("objects_config", "robotx.yaml");

    // Get parameters
    this->input_detections_topics_ =
        this->get_parameter("input_detections_topics").as_string_array();
    this->camera_info_topics_ =
        this->get_parameter("camera_info_topics").as_string_array();
    this->output_markers_topic_ = this->get_parameter("output_markers_topic").as_string();

    // Initialize publisher
    publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        output_markers_topic_, 10);

    // Subscribe to topics for camera info and 2D detections
    for (const auto& topic : input_detections_topics_)
    {
      detection_subscribers_.push_back(
          this->create_subscription<bb_perception_msgs::msg::DetectedObject2DArray>(
              topic, 10,
              std::bind(
                  &DetectedObject2DArrayVisNode::callback, this, std::placeholders::_1)));
    }
    for (const auto& topic : camera_info_topics_)
    {
      camera_info_subscribers_.push_back(
          this->create_subscription<sensor_msgs::msg::CameraInfo>(
              topic, 10,
              std::bind(
                  &DetectedObject2DArrayVisNode::cam_info_callback, this,
                  std::placeholders::_1)));
    }

    // Load config file (YAML format)
    load_config();
  }

private:
  // Load object configuration (e.g., from robotx.yaml)
  void load_config()
  {
    std::string config_path =
        ament_index_cpp::get_package_share_directory("ml_detector") +
        "/configs/objects/" + this->get_parameter("objects_config").as_string();
    try
    {
      YAML::Node config = YAML::LoadFile(config_path);
      if (config.IsNull())
      {
        RCLCPP_ERROR(this->get_logger(), "Yaml config missing!");
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

  void cam_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr cam_info_msg)
  {
    // RCLCPP_INFO(
    //     this->get_logger(), "Received CameraInfo for frame %s",
    //     cam_info_msg->header.frame_id.c_str());
    cam_info_map_[cam_info_msg->header.frame_id] = *cam_info_msg;
  }

  // Handle detection message and create markers
  void
  callback(const bb_perception_msgs::msg::DetectedObject2DArray::SharedPtr detection_msg)
  {
    visualization_msgs::msg::MarkerArray markers;
    int i = -1;
    const auto& objects = detection_msg->objects;

    for (const auto& detection : objects)
    {
      ++i;
      std::string class_name = id_to_name_[detection.hypothesis.class_id];
      visualization_msgs::msg::Marker marker;

      marker.header.frame_id = detection_msg->header.frame_id;
      marker.header.stamp = detection_msg->header.stamp;
      marker.ns = class_name;
      marker.id = i;
      marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.lifetime = rclcpp::Duration(1, 0);
      std::string fixed_frame = detection_msg->header.frame_id;
      std::string frame_v1 = "asv4/" + detection_msg->sensor.frame_id + "_optical";
      std::string frame = "";   // camera frame either xx or asv4/xx_optical
      bool found_cam_info = false;
      if (cam_info_map_.find(detection_msg->sensor.frame_id) != cam_info_map_.end())
      {
        // RCLCPP_WARN(
        //     this->get_logger(), "No CameraInfo found for frame %s",
        //     detection_msg->sensor.frame_id.c_str());   // Modify logging as needed
        found_cam_info = true;
        frame = detection_msg->sensor.frame_id;
      }
      else if (cam_info_map_.find(frame_v1) != cam_info_map_.end())
      {
        // RCLCPP_WARN(
        //     this->get_logger(), "No CameraInfo found for frame %s",
        //     detection_msg->sensor.frame_id.c_str());   // Modify logging as needed
        found_cam_info = true;
        frame = frame_v1;
      }
      if (!found_cam_info)
      {
        RCLCPP_WARN(
            this->get_logger(), "No CameraInfo found for frame %s",
            detection_msg->sensor.frame_id.c_str());   // Modify logging as needed
        continue;
      }
      auto camera_info = cam_info_map_[frame];   // Assuming sensor frame id as z value

      geometry_msgs::msg::Point ray_start;
      ray_start.x = detection_msg->sensor_pose.position.x;
      ray_start.y = detection_msg->sensor_pose.position.y;
      ray_start.z = detection_msg->sensor_pose.position.z;

      // RCLCPP_INFO(this->get_logger(), "Processing class: %s", class_name.c_str());

      auto ray_ends = calculate_rays(
          camera_info, detection.centre_x, detection.centre_y, detection.bbox_width,
          detection.bbox_height, detection_msg->sensor_pose);

      if (ray_ends.empty())
      {
        continue;
      }

      marker.points.push_back(ray_start);
      marker.points.push_back(ray_ends[0]);
      marker.points.push_back(ray_ends[1]);
      marker.points.push_back(ray_start);
      marker.points.push_back(ray_ends[3]);
      marker.points.push_back(ray_ends[2]);
      marker.points.push_back(ray_start);
      marker.points.push_back(ray_ends[0]);
      marker.points.push_back(ray_ends[1]);
      marker.points.push_back(ray_ends[3]);
      marker.points.push_back(ray_ends[2]);
      marker.points.push_back(ray_ends[0]);

      marker.scale.x = 0.01;   // Line thickness
      marker.scale.y = 0.01;
      marker.scale.z = 0.01;
      marker.color = get_color(detection.hypothesis.class_id);
      if (class_name.find("red") != std::string::npos)
      {
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
      }
      else if (class_name.find("green") != std::string::npos)
      {
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
      }
      else if (class_name.find("blue") != std::string::npos)
      {
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
      }
      marker.ns = frame + "/" + class_name;
      markers.markers.push_back(marker);

      visualization_msgs::msg::Marker text_marker;
      text_marker.header.frame_id = detection_msg->header.frame_id;
      text_marker.header.stamp = detection.hypothesis.kinematics.header.stamp;
      text_marker.ns = class_name + "/text";
      text_marker.id = i;
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.text = class_name;
      if (detection.hypothesis.track_id > 0)
      {
        text_marker.text += "(" + std::to_string(detection.hypothesis.track_id) + ")";
      }
      text_marker.pose.position = ray_ends[2];
      text_marker.scale.z = 0.2;
      text_marker.color = marker.color;
      text_marker.lifetime = rclcpp::Duration(1, 0);
      markers.markers.push_back(text_marker);
    }

    publisher_->publish(markers);
  }

  std::vector<geometry_msgs::msg::Point> calculate_rays(
      const sensor_msgs::msg::CameraInfo& camera_info, int u, int v, int w, int h,
      const geometry_msgs::msg::Pose& sensor_pose)
  {
    // Extract camera intrinsics
    double fx = camera_info.p[0];
    double fy = camera_info.p[5];
    double cx = camera_info.p[2];
    double cy = camera_info.p[6];

    // Four corners of the bounding box
    std::vector<std::pair<int, int>> bbox_corners = {
        {u - w / 2, v + h / 2},   // Bottom-left
        {u + w / 2, v + h / 2},   // Bottom-right
        {u - w / 2, v - h / 2},   // Top-left
        {u + w / 2, v - h / 2}    // Top-right
    };

    std::vector<geometry_msgs::msg::Point> ray_ends;
    std::vector<double> ts;

    for (size_t i = 0; i < bbox_corners.size(); ++i)
    {
      auto [corner_u, corner_v] = bbox_corners[i];

      // Normalized image coordinates
      double x_norm = (corner_u - cx) / fx;
      double y_norm = (corner_v - cy) / fy;

      // Ray direction assuming distance = 1 unit
      tf2::Vector3 ray_dir_camera(x_norm, y_norm, 1.0);

      // Rotate ray direction with sensor orientation
      tf2::Quaternion q;
      tf2::fromMsg(sensor_pose.orientation, q);
      tf2::Matrix3x3 rotation_matrix(q);
      tf2::Vector3 ray_dir_world = rotation_matrix * ray_dir_camera;

      // Calculate ground plane intersection (z = 0)
      double t = -sensor_pose.position.z / ray_dir_world.z();
      if (t < 0)
      {
        // RCLCPP_WARN(this->get_logger(), "Ray does not intersect ground plane");
        t = -t;
      }
      t = std::min(t, 50.0);
      ts.push_back(t);


      if (i > 1)
      {
        t = ts[i - 2];   // Use previous intersection t for consistency
      }

      geometry_msgs::msg::Point ray_end;
      ray_end.x = sensor_pose.position.x + ray_dir_world.x() * t;
      ray_end.y = sensor_pose.position.y + ray_dir_world.y() * t;
      ray_end.z = sensor_pose.position.z + ray_dir_world.z() * t;

      ray_ends.push_back(ray_end);
    }

    return ray_ends;
  }

  std_msgs::msg::ColorRGBA get_color(int id)
  {
    std_msgs::msg::ColorRGBA color;
    color.r = static_cast<float>((id >> 16) % 256) / 255.0;
    color.g = static_cast<float>((id >> 8) % 256) / 255.0;
    color.b = static_cast<float>(id % 256) / 255.0;
    color.a = 1.0;
    return color;
  }

  // Get class name from object ID
  std::string get_class_name(int class_id)
  {
    auto it = id_to_name_.find(class_id);
    return it != id_to_name_.end() ? it->second : "unknown";
  }

  std::unordered_map<int, std::string> id_to_name_;
  std::vector<std::string> input_detections_topics_, camera_info_topics_;
  std::string output_markers_topic_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr publisher_;
  std::vector<
      rclcpp::Subscription<bb_perception_msgs::msg::DetectedObject2DArray>::SharedPtr>
      detection_subscribers_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr>
      camera_info_subscribers_;
  std::unordered_map<std::string, sensor_msgs::msg::CameraInfo> cam_info_map_;
};

// int main(int argc, char** argv) {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<DetectedObject2DArrayVisNode>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }
RCLCPP_COMPONENTS_REGISTER_NODE(DetectedObject2DArrayVisNode)