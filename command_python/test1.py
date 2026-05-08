import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import Path_Planning
# 预设一个目标三维坐标 [X, Y, Z]
# 注意：目标点必须在机械臂的工作空间(Workspace)内，否则IK会解出最接近的位置
target_3d_position = np.array([0.5, 0.3, 0.25]) 
# ==========================================
# 1. 算法核心：逆运动学 (IK) 求解器
# ==========================================
def compute_ik(model, data, target_pos, site_id, max_steps=100, tol=1e-4):
    """
    使用阻尼最小二乘法(DLS)计算逆运动学
    """
    # 备份当前关节状态，防止IK计算直接污染仿真物理状态
    q_init = data.qpos.copy()
    
    for _ in range(max_steps):
        # 计算当前正运动学
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        
        # 获取当前末端执行器(site)的位置
        current_pos = data.site_xpos[site_id]
        error = target_pos - current_pos
        
        # 如果误差在容许范围内，则认为求解成功
        if np.linalg.norm(error) < tol:
            break
            
        # 获取位置雅可比矩阵 (3 x nv)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        
        # 阻尼最小二乘法求逆 (Damped Least Squares)
        damping = 1e-3
        J = jacp
        J_T = J.T
        # dq = J^T * (J * J^T + lambda^2 * I)^-1 * error
        dq = J_T @ np.linalg.inv(J @ J_T + damping**2 * np.eye(3)) @ error
        
        # 更新关节位置 (添加一个步长系数防止震荡)
        step_size = 0.5
        data.qpos[:] += dq * step_size
        
    # 保存算出的目标关节角度
    q_target = data.qpos.copy()
    # 恢复物理引擎的原始状态
    data.qpos[:] = q_init
    mujoco.mj_kinematics(model, data)
    
    return q_target

# ==========================================
# 2. 算法核心：路径规划与路点生成
# ==========================================
def generate_waypoints(model, data, target_pos, site_name="ee_site", num_steps=100):
    """
    给定目标三维坐标，生成从当前坐标到目标坐标的连续关节路点
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"在XML模型中找不到名为 '{site_name}' 的 site！请检查你的MJCF。")

    # 获取当前三维位置
    mujoco.mj_kinematics(model, data)
    start_pos = data.site_xpos[site_id].copy()
    
    print(f"当前位置: {start_pos}")
    print(f"目标位置: {target_pos}")
    
    joint_waypoints = []
    
    # 备份当前真实状态，用于后续恢复
    original_qpos = data.qpos.copy()
    
    # 笛卡尔空间线性插值
    for i in range(1, num_steps + 1):
        # 算出中间点的3D坐标
        interp_pos = start_pos + (target_pos - start_pos) * (i / num_steps)
        
        # 针对中间点计算IK
        q_waypoint = compute_ik(model, data, interp_pos, site_id)
        joint_waypoints.append(q_waypoint)
        
        # 把算出这一步的角度赋给data，作为下一步IK的初始猜测值，保证路径连续性
        data.qpos[:] = q_waypoint
        
    # 恢复原始状态
    data.qpos[:] = original_qpos
    mujoco.mj_kinematics(model, data)
    
    print(f"成功生成了 {len(joint_waypoints)} 个中间关节路点！")
    return joint_waypoints

# ==========================================
# 3. MuJoCo 仿真与主程序
# ==========================================
def main():
    # # ！！！实际使用时，请替换为你自己的模型！！！
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(current_dir, "../my_robot_description/urdf/Arm_mujoco.xml")
    # model = mujoco.MjModel.from_xml_path(model_path)
    # # 初始化仿真数据
    # data = mujoco.MjData(model)


    
    # # 替换为你XML里机械臂末端 <site> 的名字
    # end_effector_site_name = "ee_site"
    
    # # 生成路点 (切分成200步以保证运动平滑)
    # waypoints = generate_waypoints(model, data, target_3d_position, end_effector_site_name, num_steps=200)

    # # 启动交互式仿真器
    # with mujoco.viewer.launch_passive(model, data) as viewer:
    #     time.sleep(1) # 等待渲染器启动
        
    #     print("开始沿规划路点运动...")
    #     for waypoint in waypoints:
    #         if not viewer.is_running():
    #             break
                
    #         # 将规划出的关节角度路点赋给驱动器控制指令 (ctrl)
    #         data.ctrl[:] = waypoint
            
    #         # 每个路点步进多次仿真，给位置伺服足够时间驱动关节
    #         for _ in range(20):
    #             mujoco.mj_step(model, data)

    #         # 同步画面
    #         viewer.sync()

    #         # 控制运动速度 (根据你的需要调整)
    #         time.sleep(0.02)

    #     print("到达目标位置！")
        
    #     # 保持窗口打开直到手动关闭
    #     while viewer.is_running():
    #         mujoco.mj_step(model, data)
    #         viewer.sync()
    #         time.sleep(0.01)
    model_path="../my_robot_description/urdf/Arm_mujoco.xml"
    Arm_Path_Planner = Path_Planning.Arm_Path_Planner(model_path, site_name="ee_site",num_steps=100,max_steps=100, tol=1e-3)
    Arm_Path_Planner.init_sim()
    Arm_Path_Planner.set_target([0.1, 0.1, 0.25])
    Arm_Path_Planner.gogogo_sim()
    Arm_Path_Planner.get_now_position()
    # target_3d_position = np.array([0.6, 0.7, 0.4]) 
    # Arm_Path_Planner.set_target(target_3d_position)
    # Arm_Path_Planner.gogogo_sim()
    while 1:
        pass
    
if __name__ == "__main__":
    main()