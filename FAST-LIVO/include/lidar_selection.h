#ifndef LIDAR_SELECTION_H_
#define LIDAR_SELECTION_H_

#include <common_lib.h>
#include <vikit/abstract_camera.h>
#include <frame.h>
#include <map.h>
#include <feature.h>
#include <point.h>
#include <vikit/vision.h>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <set>

namespace lidar_selection {

class LidarSelector {
  public:
    int grid_size;                   // 网格大小，可能用于将激光雷达数据划分成网格以便于处理
    vk::AbstractCamera* cam;         // 指向vk::AbstractCamera类的指针，用于处理与相机相关的数据
    SparseMap* sparse_map;           // 指向SparseMap类的指针,可能用于存储稀疏地图数据
    StatesGroup* state;              // 指向StatesGroup类的指针，可能用于存储系统状态
    StatesGroup* state_propagat;     // 指向StatesGroup类的指针，可能用于存储传播后的状态
    M3D Rli, Rci, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;   // 3x3的旋转矩阵（M3D），可能用于表示不同坐标系之间的旋转关系或雅可比矩阵
    V3D Pli, Pci, Pcw;               // 3D位置向量（V3D），可能用于表示不同坐标系的原点位置
    int* align_flag;                 // 指向整数的指针，可能用于标记对齐状态
    int* grid_num;                   // 指向整数的指针，可能用于存储网格编号
    int* map_index;                  // 指向整数的指针，可能用于存储地图索引
    float* map_dist;                 // 指向浮点数的指针，可能用于存储地图中的距离数据
    float* map_value;                // 指向浮点数的指针，可能用于存储地图中的值
    float* patch_with_border_;       // 浮点数数组，可能用于存储带有边界的局部地图数据
    vector<float> patch_cache;       // 浮点数向量，可能用于缓存局部地图数据
    int width, height, grid_n_width, grid_n_height, length; // 可能用于存储图像或网格的尺寸
    SubSparseMap* sub_sparse_map;    // 指向SubSparseMap类的指针，可能用于存储子地图数据
    double fx,fy,cx,cy;              // 可能用于存储相机的内参
    bool ncc_en;                     // 可能用于启用或禁用归一化互相关
    int debug, patch_size, patch_size_total, patch_size_half; // 定义局部地图的大小
    int count_img, MIN_IMG_COUNT;   // 用于计数图像和最小图像数量
    int NUM_MAX_ITERATIONS;         // 定义最大迭代次数
    vk::robust_cost::WeightFunctionPtr weight_function_; // 计算权重函数
    float weight_scale_;           // 用于缩放权重
    double img_point_cov, outlier_threshold, ncc_thre;   // 用于存储图像点的协方差、异常值阈值和NCC阈值
    size_t n_meas_;                //!< Number of measurements
    deque< PointPtr > map_cur_frame_;     // 当前帧的地图点
    deque< PointPtr > sub_map_cur_frame_; // 子地图点
    double computeH, ekf_time;
    double ave_total = 0.0;
    int frame_count = 0;
    vk::robust_cost::ScaleEstimatorPtr scale_estimator_;

    Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
    MatrixXd H_sub, K;
    cv::flann::Index Kdtree;

    LidarSelector(const int grid_size, SparseMap* sparse_map);

    ~LidarSelector();

    void detect(cv::Mat img, PointCloudXYZI::Ptr pg);
    float CheckGoodPoints(cv::Mat img, V2D uv);
    void addFromSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg);
    void addSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg);
    void FeatureAlignment(cv::Mat img);
    void set_extrinsic(const V3D &transl, const M3D &rot);
    void init();
    void getpatch(cv::Mat img, V3D pg, float* patch_tmp, int level);
    void getpatch(cv::Mat img, V2D pc, float* patch_tmp, int level);
    void dpi(V3D p, MD(2,3)& J);
    float UpdateState(cv::Mat img, float total_residual, int level);
    double NCC(float* ref_patch, float* cur_patch, int patch_size);

    void ComputeJ(cv::Mat img);
    void reset_grid();
    void addObservation(cv::Mat img);
    void reset();
    bool initialization(FramePtr cur_frame, PointCloudXYZI::Ptr pg);   
    void createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref);
    void getWarpMatrixAffine(
      const vk::AbstractCamera& cam,
      const Vector2d& px_ref,
      const Vector3d& f_ref,
      const double depth_ref,
      const SE3& T_cur_ref,
      const int level_ref,    // px_ref对应特征点的金字塔层级
      const int pyramid_level,
      const int halfpatch_size,
      Matrix2d& A_cur_ref);
    bool align2D(
      const cv::Mat& cur_img,
      float* ref_patch_with_border,
      float* ref_patch,
      const int n_iter,
      Vector2d& cur_px_estimate,
      int index);
    void AddPoint(PointPtr pt_new);
    int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level);
    void display_keypatch(double time);
    void updateFrameState(StatesGroup state);
    V3F getpixel(cv::Mat img, V2D pc);

    void warpAffine(
      const Matrix2d& A_cur_ref,
      const cv::Mat& img_ref,
      const Vector2d& px_ref,
      const int level_ref,
      const int search_level,
      const int pyramid_level,
      const int halfpatch_size,
      float* patch);
    
    PointCloudXYZI::Ptr Map_points;
    PointCloudXYZI::Ptr Map_points_output;
    PointCloudXYZI::Ptr pg_down;
    pcl::VoxelGrid<PointType> downSizeFilter;
    unordered_map<VOXEL_KEY, VOXEL_POINTS*> feat_map;
    unordered_map<VOXEL_KEY, float> sub_feat_map; //timestamp
    unordered_map<int, Warp*> Warp_map; // reference frame id, A_cur_ref and search_level

    vector<VOXEL_KEY> occupy_postions;
    set<VOXEL_KEY> sub_postion;
    vector<PointPtr> voxel_points_;
    vector<V3D> add_voxel_points_;


    cv::Mat img_cp, img_rgb;
    cv::Mat img_rx;
    std::vector<FramePtr> overlap_kfs_;
    FramePtr new_frame_;
    FramePtr last_kf_;
    Map map_;
    enum Stage {
      STAGE_FIRST_FRAME,
      STAGE_DEFAULT_FRAME
    };
    Stage stage_;
    enum CellType {
      TYPE_MAP = 1,
      TYPE_POINTCLOUD,
      TYPE_UNKNOWN
    };

  private:
    struct Candidate 
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      PointPtr pt;       
      Vector2d px;    
      Candidate(PointPtr pt, Vector2d& px) : pt(pt), px(px) {}
    };
    typedef std::list<Candidate > Cell;
    typedef std::vector<Cell*> CandidateGrid;

    /// The grid stores a set of candidate matches. For every grid cell we try to find one match.
    struct Grid
    {
      CandidateGrid cells;
      vector<int> cell_order;
      int cell_size;
      int grid_n_cols;
      int grid_n_rows;
      int cell_length;
    };

    Grid grid_;
};
  typedef boost::shared_ptr<LidarSelector> LidarSelectorPtr;

} // namespace lidar_detection

#endif // LIDAR_SELECTION_H_