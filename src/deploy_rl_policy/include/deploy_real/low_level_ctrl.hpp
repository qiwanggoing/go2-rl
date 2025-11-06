#ifndef LOW_LEVEL_CTRL_HPP
#define LOW_LEVEL_CTRL_HPP
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/motor_state.hpp"
#include "unitree_go/msg/motor_cmd.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <chrono>
#include "common/motor_crc.h" //

using namespace std;
using namespace std::chrono;

class LowLevelControl : public rclcpp::Node
{
public:
    LowLevelControl();
    // ~LowLevelControl();
private:
    void state_callback(unitree_go::msg::LowState::SharedPtr msg);
    void joy_callback(sensor_msgs::msg::Joy::SharedPtr msg);
    
    // --- 修改：从 target_pos 切换到 target_torque ---
    void target_torque_callback(std_msgs::msg::Float32MultiArray::SharedPtr msg);

    void state_machine();
    void state_obs();
    void state_transform(vector<double> &target_angels);
    void init_cmd();
    double jointLinearInterpolation(double initPos, double targetPos, double rate);

    rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr cmd_puber_;
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr state_suber_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_suber_;
    
    // --- 修改：订阅者 ---
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr target_torque_suber_;

    // --- 保留：用于调试发布 ---
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr target_pos_puber_; 
    
    rclcpp::TimerBase::SharedPtr timer_;

    unitree_go::msg::LowCmd cmd_msg_;
    unitree_go::msg::MotorState motor[12];
    std_msgs::msg::Float32MultiArray pos_data_; // 将重用于发布力矩
    
    // --- 修改：数据存储 ---
    std_msgs::msg::Float32MultiArray rl_target_torques_;

    vector<double> kp, kd;
    vector<double> q_init_ = vector<double>(12, 0);
    vector<double> q_des_ = vector<double>(12, 0);

    // 站立姿态 (FL, FR, RL, RR 顺序)
    vector<double> standing_angels_ = {0.0, 1.05, -2.4, 0.0, 1.05, -2.4, 0.0, 1.05, -2.4, 0.0, 1.05, -2.4};
    // 卧倒姿态 (FL, FR, RL, RR 顺序)
    vector<double> laydown_angels_ = {-0.4, 1.05, -2.7, 0.4, 1.05, -2.7, -0.4, 1.05, -2.7, 0.4, 1.05, -2.7};
    
    int motion_time_ = 0;
    int rate_count_ = 0;

    // 状态标志
    bool is_standing_ = false;
    bool is_laydown_ = false;
    bool is_uncontrolled_ = true;
    bool should_stand_ = false;
    bool should_laydown_ = false;
    bool should_run_policy_ = false;
    bool recieved_data_ = false; // 现在表示是否收到了力矩数据
    bool is_simulation;
};

#endif