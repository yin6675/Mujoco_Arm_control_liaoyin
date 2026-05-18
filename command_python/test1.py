import numpy as np
import time
import Path_Planning

# ==========================================
# 1. MuJoCo 仿真与主程序
# ==========================================
def main():

    model_path="../my_robot_description/urdf/Arm_mujoco.xml"
    Arm_Path_Planner = Path_Planning.Arm_Path_Planner(
        model_path, site_name="ee_site",
        reference_site_name="reference_origin",
        num_steps=200, max_steps=200, tol=1e-3
    )
    Arm_Path_Planner.init_sim()

    # # 测试1: 正常区域，高精度（单坐标 >0.5，另一个 <0.5）
    # print("\n===== 测试1: x=0.55, y=0.3 (正常区域) =====")
    # Arm_Path_Planner.set_target([0.3, 0.3, 0.25])
    # Arm_Path_Planner.gogogo_sim()
    # Arm_Path_Planner.get_now_position()
    # time.sleep(1)

    # # 测试2: 边界区域（双坐标 >0.55，之前会乱甩）
    # print("\n===== 测试2: x=0.55, y=0.55 (边界区域) =====")
    # Arm_Path_Planner.set_target([0.55, 0.55, 0.25])
    # Arm_Path_Planner.gogogo_sim()
    # Arm_Path_Planner.get_now_position()
    # time.sleep(1)

    # # 测试3: 远边界区域（双坐标均接近极限）
    # print("\n===== 测试3: x=0.6, y=0.6 (远边界区域) =====")
    # Arm_Path_Planner.set_target([0.6, 0.6, 0.25])
    # Arm_Path_Planner.gogogo_sim()
    # Arm_Path_Planner.get_now_position()
    # time.sleep(1)

    # # 测试4: 回到安全区域
    # print("\n===== 测试4: x=0.3, y=0.3 (回到安全区域) =====")
    # Arm_Path_Planner.set_target([0.3, 0.3, 0.25])
    # Arm_Path_Planner.gogogo_sim()
    # Arm_Path_Planner.get_now_position()
    # time.sleep(1)

    while 1:
        pass

if __name__ == "__main__":
    main()
