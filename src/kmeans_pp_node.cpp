// K-Means++ with Performance Measurement
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <random>
#include <limits>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono> // [속도 측정] 헤더 추가

// ... (PCL RANSAC 헤더 등 나머지 include는 동일)
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

using InputPointT = pcl::PointXYZ;

class KMeansLidar : public rclcpp::Node
{
public:
    // ... (생성자 및 private 멤버, 다른 함수들은 이전과 동일)
    KMeansLidar(int k, int max_iter)
    : Node("kmeans_lidar_node"), k_(k), max_iter_(max_iter), processed_(false)
    {
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar/pointcloud", 10,
            std::bind(&KMeansLidar::pointcloudCallback, this, std::placeholders::_1));

        centers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/kmeans_pp/initial_centers", 10);
            
        bbox_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/kmeans_pp/cluster_bounding_boxes", 10);
            
        ground_removed_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/kmeans_pp/ground_removed_cloud", 10);

        color_map_.push_back({255, 0, 0});
        color_map_.push_back({0, 255, 0});
        color_map_.push_back({0, 0, 255});
        color_map_.push_back({255, 255, 0});
        color_map_.push_back({0, 255, 255});
        color_map_.push_back({255, 0, 255});
        color_map_.push_back({255, 165, 0});
        color_map_.push_back({128, 0, 128});
    }

private:
    int k_;
    int max_iter_;
    bool processed_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr centers_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_removed_pub_;
    std::vector<std::vector<uint8_t>> color_map_;

    void publishCenters(const Eigen::MatrixXd& centers, const std_msgs::msg::Header& header)
    {
        RCLCPP_INFO(this->get_logger(), "Publishing %ld initial center markers.", centers.rows());
        visualization_msgs::msg::MarkerArray marker_array;
        for (int i = 0; i < centers.rows(); ++i) {
            RCLCPP_INFO(this->get_logger(), "  - Initial Center #%d: [%.2f, %.2f, %.2f]", i, centers(i, 0), centers(i, 1), centers(i, 2));
            visualization_msgs::msg::Marker marker;
            marker.header = header;
            marker.ns = "initial_centers";
            marker.id = i;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = centers(i, 0);
            marker.pose.position.y = centers(i, 1);
            marker.pose.position.z = centers(i, 2);
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.3;
            marker.scale.y = 0.3;
            marker.scale.z = 0.3;
            marker.color.r = 1.0f;
            marker.color.g = 0.0f;
            marker.color.b = 0.0f;
            marker.color.a = 1.0;
            marker.lifetime = rclcpp::Duration::from_seconds(0);
            marker_array.markers.push_back(marker);
        }
        centers_pub_->publish(marker_array);
    }
    
    void publishClusterBoundingBoxes(const Eigen::MatrixXd& data, 
                                     const std::vector<int>& labels, 
                                     const std_msgs::msg::Header& header)
    {
        visualization_msgs::msg::MarkerArray bbox_array;
        std::vector<std::vector<size_t>> points_in_cluster(k_);
        for (size_t i = 0; i < labels.size(); ++i) {
            points_in_cluster[labels[i]].push_back(i);
        }
        for (int i = 0; i < k_; ++i) {
            if (points_in_cluster[i].empty()) {
                continue;
            }
            Eigen::Vector3d min_pt(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
            Eigen::Vector3d max_pt(-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max());
            for (size_t pt_idx : points_in_cluster[i]) {
                min_pt.x() = std::min(min_pt.x(), data(pt_idx, 0));
                min_pt.y() = std::min(min_pt.y(), data(pt_idx, 1));
                min_pt.z() = std::min(min_pt.z(), data(pt_idx, 2));
                max_pt.x() = std::max(max_pt.x(), data(pt_idx, 0));
                max_pt.y() = std::max(max_pt.y(), data(pt_idx, 1));
                max_pt.z() = std::max(max_pt.z(), data(pt_idx, 2));
            }
            visualization_msgs::msg::Marker box_marker;
            box_marker.header = header;
            box_marker.ns = "cluster_boxes";
            box_marker.id = i;
            box_marker.type = visualization_msgs::msg::Marker::CUBE;
            box_marker.action = visualization_msgs::msg::Marker::ADD;
            box_marker.pose.position.x = (min_pt.x() + max_pt.x()) / 2.0;
            box_marker.pose.position.y = (min_pt.y() + max_pt.y()) / 2.0;
            box_marker.pose.position.z = (min_pt.z() + max_pt.z()) / 2.0;
            box_marker.pose.orientation.w = 1.0;
            box_marker.scale.x = max_pt.x() - min_pt.x();
            box_marker.scale.y = max_pt.y() - min_pt.y();
            box_marker.scale.z = max_pt.z() - min_pt.z();
            const auto& color = color_map_[i % color_map_.size()];
            box_marker.color.r = color[0] / 255.0;
            box_marker.color.g = color[1] / 255.0;
            box_marker.color.b = color[2] / 255.0;
            box_marker.color.a = 0.5;
            box_marker.lifetime = rclcpp::Duration::from_seconds(0);
            bbox_array.markers.push_back(box_marker);
        }
        bbox_pub_->publish(bbox_array);
        RCLCPP_INFO(this->get_logger(), "Published %zu cluster bounding boxes.", bbox_array.markers.size());
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (processed_) return;
        auto callback_start = std::chrono::steady_clock::now();
        
        // ... (PCL 변환, (0,0,0) 필터링) ...
        pcl::PointCloud<InputPointT>::Ptr cloud(new pcl::PointCloud<InputPointT>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<InputPointT>::Ptr cloud_filtered(new pcl::PointCloud<InputPointT>);
        cloud_filtered->reserve(cloud->size());
        for (const auto& point : *cloud) {
            if (std::abs(point.x) > 0.01 || std::abs(point.y) > 0.01 || std::abs(point.z) > 0.01) {
                cloud_filtered->points.push_back(point);
            }
        }
        cloud_filtered->width = cloud_filtered->points.size();
        cloud_filtered->height = 1;
        cloud_filtered->is_dense = true;

        // [속도 측정] RANSAC
        auto ransac_start = std::chrono::steady_clock::now();
        pcl::PointCloud<InputPointT>::Ptr cloud_no_ground(new pcl::PointCloud<InputPointT>);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<InputPointT> seg;
        pcl::ExtractIndices<InputPointT> extract;

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.1);

        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            cloud_no_ground = cloud_filtered;
        } else {
            extract.setInputCloud(cloud_filtered);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud_no_ground);
        }
        auto ransac_end = std::chrono::steady_clock::now();

        // ...
        if (cloud_no_ground->empty() || cloud_no_ground->size() < static_cast<size_t>(k_)) {
             return;
        }

        const size_t n = cloud_no_ground->size();
        Eigen::MatrixXd data(n, 3);
        for (size_t i = 0; i < n; ++i) {
            data(i,0) = cloud_no_ground->points[i].x;
            data(i,1) = cloud_no_ground->points[i].y;
            data(i,2) = cloud_no_ground->points[i].z;
        }

        // [속도 측정] K-Means++ 초기 중심점 선택
        auto init_start = std::chrono::steady_clock::now();
        Eigen::MatrixXd centers(k_, 3);
        std::mt19937 gen(std::random_device{}());

        std::uniform_int_distribution<size_t> dist(0, n - 1);
        centers.row(0) = data.row(dist(gen));

        std::vector<double> min_sq_dists(n);
        for (int i = 1; i < k_; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double min_d = std::numeric_limits<double>::max();
                for (int c = 0; c < i; ++c) {
                    double d = (data.row(j) - centers.row(c)).squaredNorm();
                    min_d = std::min(min_d, d);
                }
                min_sq_dists[j] = min_d;
            }
            std::discrete_distribution<> dd(min_sq_dists.begin(), min_sq_dists.end());
            centers.row(i) = data.row(dd(gen));
        }
        auto init_end = std::chrono::steady_clock::now();

        publishCenters(centers, msg->header);

        // [속도 측정] K-Means 반복
        auto kmeans_start = std::chrono::steady_clock::now();
        std::vector<int> labels;
        kmeans(data, k_, max_iter_, labels, centers);
        auto kmeans_end = std::chrono::steady_clock::now();
        
        // ... (클러스터 개수 카운트 및 BBox 발행) ...
        RCLCPP_INFO(this->get_logger(), "Cluster point counts:");
        std::vector<int> cluster_counts(k_, 0);
        for (int label : labels) {
            if (label >= 0 && label < k_) {
                cluster_counts[label]++;
            }
        }
        for (int i = 0; i < k_; ++i) {
            RCLCPP_INFO(this->get_logger(), "  - Cluster #%d: %d points", i, cluster_counts[i]);
        }
        
        publishClusterBoundingBoxes(data, labels, msg->header);
        
        processed_ = true;
        auto callback_end = std::chrono::steady_clock::now();

        // [속도 측정] 최종 결과 출력
        auto ransac_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ransac_end - ransac_start);
        auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);
        auto kmeans_duration = std::chrono::duration_cast<std::chrono::milliseconds>(kmeans_end - kmeans_start);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(callback_end - callback_start);

        RCLCPP_INFO(this->get_logger(), "--- Performance Analysis (K-Means++) ---");
        RCLCPP_INFO(this->get_logger(), "RANSAC Ground Removal: %ld ms", ransac_duration.count());
        RCLCPP_INFO(this->get_logger(), "Centroid Initialization: %ld ms", init_duration.count());
        RCLCPP_INFO(this->get_logger(), "K-Means Iteration:     %ld ms", kmeans_duration.count());
        RCLCPP_INFO(this->get_logger(), "Total Callback Time:     %ld ms", total_duration.count());
        RCLCPP_INFO(this->get_logger(), "----------------------------------------");
    }

    void kmeans(const Eigen::MatrixXd &data, int k, int max_iter,
                std::vector<int> &labels, Eigen::MatrixXd &centers)
    {
        // ... (이 함수 내용은 변경 없음)
        size_t n = data.rows();
        size_t dim = data.cols();
        labels.assign(n, 0);
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        for (int iter = 0; iter < max_iter; ++iter) {
            bool changed = false;
            for (size_t i = 0; i < n; ++i) {
                double best_dist = std::numeric_limits<double>::max();
                int best_k = 0;
                for (int c = 0; c < k; ++c) {
                    double d = (data.row(i) - centers.row(c)).squaredNorm();
                    if (d < best_dist) {
                        best_dist = d;
                        best_k = c;
                    }
                }
                if (labels[i] != best_k) {
                    labels[i] = best_k;
                    changed = true;
                }
            }
            Eigen::MatrixXd new_centers = Eigen::MatrixXd::Zero(k, dim);
            std::vector<int> counts(k, 0);
            for (size_t i = 0; i < n; ++i) {
                new_centers.row(labels[i]) += data.row(i);
                counts[labels[i]]++;
            }
            for (int c = 0; c < k; ++c) {
                if (counts[c] > 0)
                    new_centers.row(c) /= counts[c];
                else
                  new_centers.row(c) = data.row(dist(gen));
            }
            if (!changed) {
                RCLCPP_INFO(this->get_logger(), "K-Means converged after %d iterations.", iter + 1);
                break;
            }
            centers = new_centers;
        }
    }
};

// ... (main 함수는 변경 없음)
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KMeansLidar>(5 /*k*/, 20 /*max_iter*/);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}