#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>

class LocalCloudCropperNode : public rclcpp::Node
{
public:
  LocalCloudCropperNode() : Node("local_cloud_cropper_node")
  {
    crop_x_half_ = this->declare_parameter<double>("crop_x_half", 4.0);
    crop_y_half_ = this->declare_parameter<double>("crop_y_half", 4.0);
    crop_z_down_ = this->declare_parameter<double>("crop_z_down", 1.0);
    crop_z_up_ = this->declare_parameter<double>("crop_z_up", 3.0);
    min_points_warn_threshold_ =
      this->declare_parameter<int>("min_points_warn_threshold", 20);

    cloud_topic_ = this->declare_parameter<std::string>(
      "cloud_topic", "/voxl_mapper_aligned_ptcloud");
    odom_topic_ = this->declare_parameter<std::string>(
      "odom_topic", "/fmu/out/vehicle_odometry");
    output_topic_ = this->declare_parameter<std::string>(
      "output_topic", "/starling/local_cloud");

    using std::placeholders::_1;

    // Explicit QoS instead of SensorDataQoS to reduce VOXL/PX4 weirdness.
    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_,
      sub_qos,
      std::bind(&LocalCloudCropperNode::cloudCallback, this, _1));

    odom_sub_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
      odom_topic_,
      sub_qos,
      std::bind(&LocalCloudCropperNode::odomCallback, this, _1));

    // Keep output publisher simple and reliable for easier inspection.
    cropped_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_,
      rclcpp::QoS(10));

    heartbeat_timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&LocalCloudCropperNode::heartbeatCallback, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Local cloud cropper started. cloud='%s', odom='%s', output='%s'",
      cloud_topic_.c_str(), odom_topic_.c_str(), output_topic_.c_str());
  }

private:
  void odomCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
  {
    odom_count_++;

    current_x_ = static_cast<double>(msg->position[0]);
    current_y_ = static_cast<double>(msg->position[1]);
    current_z_ = static_cast<double>(msg->position[2]);
    have_pose_ = true;
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    cloud_count_++;

    if (!have_pose_) {
      RCLCPP_WARN(
        this->get_logger(),
        "Cloud received but no odometry yet. Skipping crop.");
      return;
    }

    if (msg->height == 0 || msg->width == 0) {
      RCLCPP_WARN(
        this->get_logger(),
        "Received empty PointCloud2.");
      return;
    }

    const double x_min = current_x_ - crop_x_half_;
    const double x_max = current_x_ + crop_x_half_;
    const double y_min = current_y_ - crop_y_half_;
    const double y_max = current_y_ + crop_y_half_;
    const double z_min = current_z_ - crop_z_down_;
    const double z_max = current_z_ + crop_z_up_;

    std::vector<float> kept_xyz;
    kept_xyz.reserve(
      static_cast<std::size_t>(msg->width) *
      static_cast<std::size_t>(msg->height) * 3U);

    try {
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

      const std::size_t n_points =
        static_cast<std::size_t>(msg->width) *
        static_cast<std::size_t>(msg->height);

      for (std::size_t i = 0; i < n_points; ++i, ++iter_x, ++iter_y, ++iter_z) {
        const float x = *iter_x;
        const float y = *iter_y;
        const float z = *iter_z;

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }

        if (x >= x_min && x <= x_max &&
            y >= y_min && y <= y_max &&
            z >= z_min && z <= z_max) {
          kept_xyz.push_back(x);
          kept_xyz.push_back(y);
          kept_xyz.push_back(z);
        }
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to parse PointCloud2 fields: %s", e.what());
      return;
    }

    sensor_msgs::msg::PointCloud2 out_msg;
    out_msg.header = msg->header;
    out_msg.height = 1;
    out_msg.width = static_cast<std::uint32_t>(kept_xyz.size() / 3U);
    out_msg.is_bigendian = false;
    out_msg.is_dense = false;

    sensor_msgs::PointCloud2Modifier modifier(out_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(out_msg.width);

    sensor_msgs::PointCloud2Iterator<float> out_x(out_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(out_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(out_msg, "z");

    for (std::size_t i = 0; i < kept_xyz.size(); i += 3) {
      *out_x = kept_xyz[i];
      *out_y = kept_xyz[i + 1];
      *out_z = kept_xyz[i + 2];
      ++out_x;
      ++out_y;
      ++out_z;
    }

    cropped_pub_->publish(out_msg);
    published_count_++;

    const std::size_t kept_points = kept_xyz.size() / 3U;

    if (static_cast<int>(kept_points) < min_points_warn_threshold_) {
      RCLCPP_WARN(
        this->get_logger(),
        "Cropped cloud sparse: %zu points. Pose=(%.3f, %.3f, %.3f)",
        kept_points, current_x_, current_y_, current_z_);
    } else {
      RCLCPP_INFO(
        this->get_logger(),
        "Published cropped cloud with %zu points. Pose=(%.3f, %.3f, %.3f)",
        kept_points, current_x_, current_y_, current_z_);
    }
  }

  void heartbeatCallback()
  {
    RCLCPP_INFO(
      this->get_logger(),
      "Heartbeat: odom_count=%zu cloud_count=%zu published_count=%zu have_pose=%s pose=(%.3f, %.3f, %.3f)",
      odom_count_,
      cloud_count_,
      published_count_,
      have_pose_ ? "true" : "false",
      current_x_, current_y_, current_z_);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cropped_pub_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;

  std::string cloud_topic_;
  std::string odom_topic_;
  std::string output_topic_;

  double crop_x_half_{4.0};
  double crop_y_half_{4.0};
  double crop_z_down_{1.0};
  double crop_z_up_{3.0};
  int min_points_warn_threshold_{20};

  bool have_pose_{false};
  double current_x_{0.0};
  double current_y_{0.0};
  double current_z_{0.0};

  std::size_t odom_count_{0};
  std::size_t cloud_count_{0};
  std::size_t published_count_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalCloudCropperNode>());
  rclcpp::shutdown();
  return 0;
}
