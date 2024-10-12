#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "bb_perception_msgs/msg/detected_object2_d_array.hpp"
#include "bb_perception_msgs/msg/detected_object3_d_array.hpp"
#include "bb_perception_msgs/msg/detected_object2_d.hpp"
#include "bb_perception_msgs/msg/detected_object3_d.hpp"
#include "bb_perception_msgs/msg/object_classification.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using std::placeholders::_1;
using bb_perception_msgs::msg::DetectedObject2DArray;
using bb_perception_msgs::msg::DetectedObject3DArray;
using bb_perception_msgs::msg::DetectedObject2D;
using bb_perception_msgs::msg::DetectedObject3D;
using bb_perception_msgs::msg::ObjectClassification;
using sensor_msgs::msg::CameraInfo;
using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::Pose;

class DetectedObject3DLabelingNode : public rclcpp::Node {
public:
    DetectedObject3DLabelingNode()
        : Node("detected_object_3d_labelling_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {

        declare_parameter<std::string>("detection_2d_topic", "/asv4/vision/detections_2d");
        declare_parameter<std::string>("detection_3d_topic", "/asv4/vision/lidar_small_objects/dets_3d/filtered");
        declare_parameter<std::vector<std::string>>("camera_info_topics", {"/asv4/left_cam/camera_info", "/asv4/right_cam/camera_info", "/asv4/front_cam/camera_info"});
        declare_parameter<std::string>("output_labeled_topic", "/asv4/vision/lidar_small_objects/dets_3d/labelled");

        get_parameter("detection_2d_topic", detection_2d_topic_);
        get_parameter("detection_3d_topic", detection_3d_topic_);
        get_parameter("camera_info_topics", camera_info_topics_);
        get_parameter("output_labeled_topic", output_labeled_topic_);

        for (const auto & topic : camera_info_topics_) {
            camera_info_subscribers_.emplace_back(create_subscription<CameraInfo>(topic, 10, std::bind(&DetectedObject3DLabelingNode::camera_info_callback, this, _1)));
        }

        detection_2d_subscriber_ = create_subscription<DetectedObject2DArray>(
            detection_2d_topic_, 10, std::bind(&DetectedObject3DLabelingNode::detection_2d_callback, this, _1));

        detection_3d_subscriber_ = create_subscription<DetectedObject3DArray>(
            detection_3d_topic_, 10, std::bind(&DetectedObject3DLabelingNode::detection_3d_callback, this, _1));

        labeled_3d_publisher_ = create_publisher<DetectedObject3DArray>(output_labeled_topic_, 10);

        inflate_width_ = 1.5;
    }

private:
    void camera_info_callback(const CameraInfo::SharedPtr camera_info) {
        camera_info_dict_[camera_info->header.frame_id] = *camera_info;
    }

    void detection_3d_callback(const DetectedObject3DArray::SharedPtr detection_3d_msg) {
        latest_3d_detections_ = detection_3d_msg;
    }

    void detection_2d_callback(const DetectedObject2DArray::SharedPtr detection_2d_msg) {
        if (!latest_3d_detections_) {
            return;
        }

        const auto & camera_info = camera_info_dict_.find(detection_2d_msg->sensor.frame_id);
        if (camera_info == camera_info_dict_.end()) {
            RCLCPP_WARN(this->get_logger(), "No CameraInfo for frame %s", detection_2d_msg->sensor.frame_id.c_str());
            return;
        }

        DetectedObject3DArray labeled_3d_objects;
        labeled_3d_objects.header = latest_3d_detections_->header;
        labeled_3d_objects.name = latest_3d_detections_->name;
        labeled_3d_objects.source = latest_3d_detections_->source;
        labeled_3d_objects.sensor_pose = latest_3d_detections_->sensor_pose;

        std::vector<std::vector<double>> cost_matrix(latest_3d_detections_->objects.size(), std::vector<double>(detection_2d_msg->objects.size(), 1e9));

        for (size_t i = 0; i < latest_3d_detections_->objects.size(); ++i) {
            const auto projected_2d_points = project_3d_to_2d(camera_info->second, detection_2d_msg->header, detection_2d_msg->sensor_pose, latest_3d_detections_->objects[i]);

            if (projected_2d_points.empty()) {
                continue;
            }

            for (size_t j = 0; j < detection_2d_msg->objects.size(); ++j) {
                const auto det_bbox = get_bbox_from_2d_detection(detection_2d_msg->objects[j]);
                const auto proj_bbox = get_bbox_from_2d_points(projected_2d_points);
                double overlap = compute_overlap(det_bbox, proj_bbox);
                cost_matrix[i][j] = 1 / (overlap + 1e-9);
            }
        }

        if (cost_matrix.empty()) {
            return;
        }

        auto [assignments_row, assignments_col] = linear_sum_assignment(cost_matrix);

        for (size_t row = 0; row < assignments_row.size(); ++row) {
            size_t i = assignments_row[row];
            size_t j = assignments_col[row];

            auto & track_id = latest_3d_detections_->objects[i].hypothesis.track_id;
            auto class_id = detection_2d_msg->objects[j].hypothesis.class_id;
            track_identities_[track_id][class_id] += 1;
        }

        label_3d_objects(labeled_3d_objects);
        labeled_3d_publisher_->publish(labeled_3d_objects);
    }

    void label_3d_objects(DetectedObject3DArray & labeled_3d_objects) {
        for (auto & obj_3d : latest_3d_detections_->objects) {
            auto & track_counts = track_identities_[obj_3d.hypothesis.track_id];
            if (!track_counts.empty()) {
                auto most_common_class = std::max_element(track_counts.begin(), track_counts.end(), [](const auto & a, const auto & b) {
                    return a.second < b.second;
                });
                obj_3d.hypothesis.class_id = most_common_class->first;

                labeled_3d_objects.objects.push_back(obj_3d);
            }
        }
    }

    std::vector<std::pair<double, double>> project_3d_to_2d(const CameraInfo & camera_info, const std_msgs::msg::Header & header, const Pose & sensor_pose, const DetectedObject3D & obj_3d) {
        // Perform 3D to 2D projection
        try {
            auto transform = tf_buffer_.lookupTransform(camera_info.header.frame_id, obj_3d.hypothesis.kinematics.header.frame_id, rclcpp::Time(0), rclcpp::Duration(0.1));
            PoseStamped pose_stamped;
            pose_stamped.header = obj_3d.hypothesis.kinematics.header;
            pose_stamped.pose = obj_3d.hypothesis.kinematics.pose_with_covariance.pose;

            Pose transformed_pose = tf2::doTransform(pose_stamped.pose, transform);

            // Only project objects in front of the camera
            if (transformed_pose.position.z <= 0) {
                return {};
            }

            // Compute 2D projection
            double fx = camera_info.p[0];
            double fy = camera_info.p[5];
            double cx = camera_info.p[2];
            double cy = camera_info.p[6];

            double u = (transformed_pose.position.x * fx / transformed_pose.position.z) + cx;
            double v = (transformed_pose.position.y * fy / transformed_pose.position.z) + cy;

            double bbox_width = (std::max(obj_3d.hypothesis.shape.dimensions.x, obj_3d.hypothesis.shape.dimensions.y) + inflate_width_) / transformed_pose.position.z * fx;
            double bbox_height = (obj_3d.hypothesis.shape.dimensions.z + inflate_width_) / transformed_pose.position.z * fy;

            return {
                {u - bbox_width / 2, v - bbox_height / 2},
                {u + bbox_width / 2, v - bbox_height / 2},
                {u + bbox_width / 2, v + bbox_height / 2},
                {u - bbox_width / 2, v + bbox_height / 2}
            };
        } catch (const std::exception & e) {
            RCLCPP_WARN(this->get_logger(), "Failed to lookup transform: %s", e.what());
            return {};
        }
    }

    std::pair<double, double> get_bbox_from_2d_detection(const DetectedObject2D & obj_2d) {
        double xmin = obj_2d.bbox.center.position.x - obj_2d.bbox.size_x / 2;
        double xmax = obj_2d.bbox.center.position.x + obj_2d.bbox.size_x / 2;
        double ymin = obj_2d.bbox.center.position.y - obj_2d.bbox.size_y / 2;
        double ymax = obj_2d.bbox.center.position.y + obj_2d.bbox.size_y / 2;
        return {xmin, xmax, ymin, ymax};
    }

    std::pair<double, double> get_bbox_from_2d_points(const std::vector<std::pair<double, double>> & points) {
        double xmin = std::min_element(points.begin(), points.end(), [](const auto & a, const auto & b) { return a.first < b.first; })->first;
        double xmax = std::max_element(points.begin(), points.end(), [](const auto & a, const auto & b) { return a.first > b.first; })->first;
        double ymin = std::min_element(points.begin(), points.end(), [](const auto & a, const auto & b) { return a.second < b.second; })->second;
        double ymax = std::max_element(points.begin(), points.end(), [](const auto & a, const auto & b) { return a.second > b.second; })->second;
        return {xmin, xmax, ymin, ymax};
    }

    double compute_overlap(const std::pair<double, double> & bbox1, const std::pair<double, double> & bbox2) {
        double intersect_width = std::max(0.0, std::min(bbox1.second, bbox2.second) - std::max(bbox1.first, bbox2.first));
        double intersect_height = std::max(0.0, std::min(bbox1.fourth, bbox2.fourth) - std::max(bbox1.third, bbox2.third));
        double intersection_area = intersect_width * intersect_height;
        double bbox1_area = (bbox1.second - bbox1.first) * (bbox1.fourth - bbox1.third);
        double bbox2_area = (bbox2.second - bbox2.first) * (bbox2.fourth - bbox2.third);
        double union_area = bbox1_area + bbox2_area - intersection_area;
        return intersection_area / union_area;
    }

    // Fields
    std::string detection_2d_topic_;
    std::string detection_3d_topic_;
    std::vector<std::string> camera_info_topics_;
    std::string output_labeled_topic_;
    double inflate_width_;
    DetectedObject3DArray::SharedPtr latest_3d_detections_;

    rclcpp::Subscription<DetectedObject2DArray>::SharedPtr detection_2d_subscriber_;
    rclcpp::Subscription<DetectedObject3DArray>::SharedPtr detection_3d_subscriber_;
    rclcpp::Publisher<DetectedObject3DArray>::SharedPtr labeled_3d_publisher_;

    std::vector<rclcpp::Subscription<CameraInfo>::SharedPtr> camera_info_subscribers_;
    std::unordered_map<std::string, CameraInfo> camera_info_dict_;

    std::unordered_map<int32_t, std::unordered_map<int32_t, int>> track_identities_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DetectedObject3DLabelingNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
