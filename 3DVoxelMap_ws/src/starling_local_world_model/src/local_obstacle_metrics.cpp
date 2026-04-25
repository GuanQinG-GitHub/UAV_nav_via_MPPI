#include <cmath>
#include <cfloat>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

class LocalObstacleMetricsNode : public rclcpp::Node
{
public:
  LocalObstacleMetricsNode() : Node("local_obstacle_metrics_node")
  {
    input_topic_ = this->declare_parameter<std::string>(
      "input_topic", "/starling/local_cloud");
    output_topic_ = this->declare_parameter<std::string>(
      "output_topic", "/starling/obstacle_metrics");

    max_considered_distance_ = this->declare_parameter<double>(
      "max_considered_distance", 8.0);

    forward_x_min_ = this->declare_parameter<double>(
      "forward_x_min", 0.1);
    lateral_split_threshold_ = this->declare_parameter<double>(
      "lateral_split_threshold", 0.15);
    vertical_split_threshold_ = this->declare_parameter<double>(
      "vertical_split_threshold", 0.15);

    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();

    using std::placeholders::_1;

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_,
      sub_qos,
      std::bind(&LocalObstacleMetricsNode::cloudCallback, this, _1));

    metrics_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      output_topic_,
      rclcpp::QoS(10));

    RCLCPP_INFO(
      this->get_logger(),
      "Local obstacle metrics node started. input='%s', output='%s'",
      input_topic_.c_str(), output_topic_.c_str());
  }

private:
  static float finiteOrMax(double value, double max_value)
  {
    if (!std::isfinite(value)) {
      return static_cast<float>(max_value);
    }
    return static_cast<float>(value);
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (msg->width == 0 || msg->height == 0) {
      RCLCPP_WARN(this->get_logger(), "Received empty local cloud.");
      return;
    }

    double overall_min = std::numeric_limits<double>::infinity();
    double front_min = std::numeric_limits<double>::infinity();
    double left_min = std::numeric_limits<double>::infinity();
    double right_min = std::numeric_limits<double>::infinity();
    double up_min = std::numeric_limits<double>::infinity();
    double down_min = std::numeric_limits<double>::infinity();

    std::size_t valid_points = 0;

    try {
      sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

      const std::size_t n_points =
        static_cast<std::size_t>(msg->width) *
        static_cast<std::size_t>(msg->height);

      for (std::size_t i = 0; i < n_points; ++i, ++iter_x, ++iter_y, ++iter_z) {
        const double x = static_cast<double>(*iter_x);
        const double y = static_cast<double>(*iter_y);
        const double z = static_cast<double>(*iter_z);

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
          continue;
        }

        const double dist = std::sqrt(x * x + y * y + z * z);

        if (!std::isfinite(dist) || dist > max_considered_distance_) {
          continue;
        }

        valid_points++;

        if (dist < overall_min) {
          overall_min = dist;
        }

        // Front sector
        if (x >= forward_x_min_ && dist < front_min) {
          front_min = dist;
        }

        // Left and right sectors
        if (y >= lateral_split_threshold_ && dist < left_min) {
          left_min = dist;
        }
        if (y <= -lateral_split_threshold_ && dist < right_min) {
          right_min = dist;
        }

        // Up and down sectors
        if (z >= vertical_split_threshold_ && dist < up_min) {
          up_min = dist;
        }
        if (z <= -vertical_split_threshold_ && dist < down_min) {
          down_min = dist;
        }
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to parse PointCloud2 fields: %s", e.what());
      return;
    }

    std_msgs::msg::Float32MultiArray out_msg;
    out_msg.data.resize(6);

    out_msg.data[0] = finiteOrMax(overall_min, max_considered_distance_);
    out_msg.data[1] = finiteOrMax(front_min, max_considered_distance_);
    out_msg.data[2] = finiteOrMax(left_min, max_considered_distance_);
    out_msg.data[3] = finiteOrMax(right_min, max_considered_distance_);
    out_msg.data[4] = finiteOrMax(up_min, max_considered_distance_);
    out_msg.data[5] = finiteOrMax(down_min, max_considered_distance_);

    metrics_pub_->publish(out_msg);

    RCLCPP_INFO(
      this->get_logger(),
      "Obstacle metrics: overall=%.3f front=%.3f left=%.3f right=%.3f up=%.3f down=%.3f valid_points=%zu",
      out_msg.data[0],
      out_msg.data[1],
      out_msg.data[2],
      out_msg.data[3],
      out_msg.data[4],
      out_msg.data[5],
      valid_points);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr metrics_pub_;

  std::string input_topic_;
  std::string output_topic_;

  double max_considered_distance_{8.0};
  double forward_x_min_{0.1};
  double lateral_split_threshold_{0.15};
  double vertical_split_threshold_{0.15};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalObstacleMetricsNode>());
  rclcpp::shutdown();
  return 0;
}
