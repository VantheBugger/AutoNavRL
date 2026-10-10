import numpy as np
import random
import shapely
from irsim.lib.handler.geometry_handler import GeometryFactory
from irsim.env import EnvBase


class SIM_ENV:
    """
    Simulator environment wrapper for robot navigation.
    
    This class provides a high-level interface for robot navigation simulation,
    handling state tracking, reward calculation, and environment interactions.
        """
        
    def __init__(self, world_file="robot_world.yaml", render=False):
        """
        Initialize the simulation environment.
        
        Args:
            world_file (str): Path to the world configuration file
            render (bool): Whether to enable visualization
        """
        # Initialize environment
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render)
        self.robot_goal = self.env.get_robot_info(0).goal
        self.action_dim = self._infer_action_dim()
        
        # Initialize tracking variables
        self._reset_tracking()

        # 新增：用于存储画布上的箭头和预测点对象
        self.goal_arrow_obj = None

    def _infer_action_dim(self):
        """Infer the command dimension from IRSim's velocity shape."""
        vel_shape = getattr(self.env.robot, "vel_shape", None)
        if vel_shape is not None:
            return int(vel_shape[0])

        vel_max = getattr(self.env.robot, "vel_max", None)
        if vel_max is not None:
            return int(np.asarray(vel_max).reshape(-1).size)

        return 2

    def _reset_tracking(self):
        """Reset tracking variables for distance and angle differences."""
        self.prev_distance = None
        self.prev_diff_rad = None
        self.prev_scan = None  # 新增：缓存上一帧雷达数据

    def _calculate_robot_metrics(self, robot_state):
        """
        Calculate robot-related metrics including goal vector, distance, and orientation.
        
        Args:
            robot_state: Current state of the robot [x, y, theta]
            
        Returns:
            tuple: (goal_vector, distance, cos, sin, diff_rad)
        """
        # Normalize shaped arrays (e.g. (1,), (1,1)) into scalars for compatibility
        robot_state = np.asarray(robot_state, dtype=float).reshape(-1)
        goal_state = np.asarray(self.env.robot.goal, dtype=float).reshape(-1)
        robot_goal = np.asarray(self.robot_goal, dtype=float).reshape(-1)

        # Calculate goal vector
        goal_vector = [
            robot_goal[0] - robot_state[0],
            robot_goal[1] - robot_state[1],
        ]
        
        # Calculate angle difference between robot orientation and goal orientation
        diff_rad = float(((-robot_state[2] + goal_state[2] + np.pi) % (2 * np.pi)) - np.pi)
        
        # Calculate distance and pose
        distance = np.linalg.norm(goal_vector)
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self._calculate_cossin(pose_vector, goal_vector)
        
        return goal_vector, distance, cos, sin, diff_rad

    @staticmethod
    def _calculate_cossin(vec1, vec2):
        """
        Calculate cosine and sine between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            tuple: (cosine, sine) of the angle between vectors
        """
        vec1 = np.asarray(vec1, dtype=float)
        vec2 = np.asarray(vec2, dtype=float)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin

    def _calculate_reward(self, goal, collision, distance_delta, action, laser_scan, delta_rad, scan_velocity, pvo_collision_prob=0):
        """
        Calculate reward based on various factors including goal achievement,
        collision avoidance, and movement efficiency.
        
        Args:
            goal (bool): Whether goal is reached
            collision (bool): Whether collision occurred
            distance_delta (float): Change in distance to goal
            action (list): [linear_velocity, angular_velocity]
            laser_scan (list): Laser scan readings
            delta_rad (float): Change in angle difference
            
        Returns:
            float: Calculated reward
        """
        if goal:
            return 100.0
        elif collision:
            return -100.0
        
        # Reward components
        progress_reward = distance_delta * 10  # Reward for moving closer to goal
        dir_progress = delta_rad * 1  # Reward for aligning with goal
        time_penalty = -0.65  # Penalty for each time step
        yaw_rate = action[2] if len(action) >= 3 else action[1]
        lateral_velocity = action[1] if len(action) >= 3 else 0.0
        rotation_penalty = -abs(yaw_rate) * 0.4  # Penalty for excessive rotation
        lateral_penalty = -abs(lateral_velocity) * 0.05
        
        # Obstacle avoidance
        safe_distance = 1.35
        min_dist = min(laser_scan)
        obstacle_penalty = -(safe_distance - min_dist) if min_dist < safe_distance else 0
        
        # 3. 新增：动态接近惩罚 (Approach Penalty)
        # 如果 scan_velocity 为负，说明障碍物在靠近。我们惩罚那种“迎头撞上”的趋势。
        approach_penalty = 0
        if scan_velocity is not None:
            # 找到所有雷达束中，障碍物靠近最快的那一个值
            # 负值越小，代表靠近越快
            fastest_approach = min(scan_velocity) 
            if fastest_approach < -0.05: # 设置一个微小的触发阈值
                # 惩罚项 = 靠近速度 * 权重
                approach_penalty = fastest_approach * 1

        pvo_penalty = -2.0 * pvo_collision_prob

        return progress_reward + time_penalty + obstacle_penalty + dir_progress + rotation_penalty + lateral_penalty + approach_penalty + pvo_penalty

    def step(self, lin_velocity=0.0, ang_velocity=0.1, lateral_velocity=0.0):
        """
        Execute one simulation step.
        
        Args:
            lin_velocity (float): Forward velocity
            ang_velocity (float): Angular velocity
            lateral_velocity (float): Body-frame lateral velocity for omni-angular/4WIS simulation
            
        Returns:
            tuple: (laser_scan, distance, cos, sin, collision, goal, diff_rad, action, reward)
        """
        # Step simulation
        if self.action_dim >= 3:
            sim_action = np.array([[lin_velocity], [lateral_velocity], [ang_velocity]])
            action = [lin_velocity, lateral_velocity, ang_velocity]
        else:
            sim_action = np.array([[lin_velocity], [ang_velocity]])
            action = [lin_velocity, ang_velocity]

        self.env.step(action_id=0, action=sim_action)
        if self.env.display:
            self.env.render()
            self._plot_goal_arrow()
            # self._plot_prediction()

        # Get sensor data
        scan = self.env.get_lidar_scan()
        current_scan = np.array(scan["ranges"], dtype=np.float32) # 只取数值列表并转为浮点数组
        # 处理无穷远
        inf_mask = np.isinf(current_scan)
        current_scan[inf_mask] = 8.0
        
        # --- 计算 scan_velocity ---
        if self.prev_scan is None:
            scan_velocity = np.zeros_like(current_scan)
        else:
            # 变化率 = 当前距离 - 上一帧距离
            # 负值表示障碍物在靠近，正值表示在远离
            scan_velocity = current_scan - self.prev_scan
        
        # 更新缓存
        self.prev_scan = current_scan.copy()
        # --- 结束计算 ---
        robot_state = self.env.get_robot_state()
        
        # Calculate metrics
        goal_vector, distance, cos, sin, diff_rad = self._calculate_robot_metrics(robot_state)
        
        # Calculate deltas
        if self.prev_distance is None:
            distance_delta = delta_rad = 0
        else:
            distance_delta = self.prev_distance - distance
            delta_rad = abs(self.prev_diff_rad) - abs(diff_rad)
        
        # Update tracking
        self.prev_distance = distance
        self.prev_diff_rad = diff_rad
        
        # Get status and calculate reward
        goal = self.env.robot.arrive
        collision = self.env.robot.collision
        reward = self._calculate_reward(goal, collision, distance_delta, action, scan["ranges"], delta_rad, scan_velocity)
        
        if goal:
            print("Goal reached")

        return scan["ranges"], distance, cos, sin, collision, goal, diff_rad, action, reward

    def _plot_goal_arrow(self):
        """
        在目标点位置绘制一个红色箭头，指向目标要求的朝向。
        """
        try:
            import matplotlib.pyplot as plt

            if self.goal_arrow_obj is not None:
                self.goal_arrow_obj.remove()
                self.goal_arrow_obj = None
            
            # 获取目标位姿 [x, y, theta]
            # 注意：根据你之前的代码，目标存储在 self.robot_goal 中
            gx = self.robot_goal[0].item()
            gy = self.robot_goal[1].item()
            gtheta = self.robot_goal[2].item()
            
            # 计算箭头的分量 (长度设为 0.5 米)
            arrow_len = 0.5
            dx = arrow_len * np.cos(gtheta)
            dy = arrow_len * np.sin(gtheta)
            
            # 在当前的 axes 上绘图
            ax = plt.gca()
            
            # 绘制箭头 (红色，起点在目标中心)
            # fc: face color, ec: edge color
            self.goal_arrow_obj = ax.arrow(gx, gy, dx, dy, width=0.05, head_width=0.2, fc='r', ec='r', zorder=10)
            
        except Exception as e:
            # 防止绘图逻辑报错影响主仿真循环
            pass

    def _plot_prediction(self, r_x_pred, r_y_pred):
        import matplotlib.pyplot as plt
        ax = plt.gca()
        # 画一个蓝色半透明小点，表示机器人预测会走到的位置
        ax.plot(r_x_pred, r_y_pred, 'bo', alpha=0.3)        

    def reset(self, robot_state=None, robot_goal=None, random_obstacles=True):
        """
        Reset the simulation environment.
        
        Args:
            robot_state (list): Initial robot state [x, y, theta]
            robot_goal (list): Goal position [x, y, theta]
            random_obstacles (bool): Whether to place random obstacles
            
        Returns:
            tuple: Initial state information
        """
        # Initialize robot state
        if robot_state is None:
            robot_state = [[random.uniform(0.5, 5.5)], 
                          [random.uniform(0.5, 5.5)], 
                          [0]]

        self.env.robot.set_state(state=np.array(robot_state), init=True)

        # Place obstacles
        if random_obstacles:
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[6, 6, 3.14],
                ids=list(range(1, 7)),
                non_overlapping=True
            )

        # Set goal
        if robot_goal is None:
            robot_goal = self._generate_valid_goal()
        
        self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal
        self._reset_tracking()
        
        # Get initial state
        action = [0.0, 0.0]
        return self.step(lin_velocity=action[0], ang_velocity=action[1])

    def _generate_valid_goal(self):
        """
        Generate a valid goal position that doesn't overlap with obstacles.
        
        Returns:
            list: Valid goal position [x, y, theta]
        """
        while True:
            goal = [[random.uniform(0.5, 5.5)], 
                   [random.uniform(0.5, 5.5)], 
                   [random.uniform(-3.14, 3.14)]]
            
            # Check if goal overlaps with obstacles
            shape = {"name": "rectangle", "length": 0.85, "width": 0.65}
            state = [goal[0], goal[1], goal[2]]
            gf = GeometryFactory.create_geometry(**shape)
            geometry = gf.step(np.c_[state])
            
            if not any(shapely.intersects(geometry, obj._geometry) for obj in self.env.obstacle_list):
                return goal
