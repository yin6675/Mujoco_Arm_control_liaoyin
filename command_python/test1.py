import numpy as np
import time
import Path_Planning

# ==========================================
# 1. MuJoCo 仿真与主程序
# ==========================================
def main():

    model_path="../my_robot_description/urdf/Arm_mujoco.xml"
    Arm_Path_Planner = Path_Planning.Arm_Path_Planner(model_path, site_name="ee_site",num_steps=100,max_steps=100, tol=1e-3)
    Arm_Path_Planner.init_sim()
    Arm_Path_Planner.set_target([0.3, 0.3, 0.25])
    Arm_Path_Planner.gogogo_sim()
    Arm_Path_Planner.get_now_position()
    time.sleep(1)
    Arm_Path_Planner.set_target([0.4, 0.4, 0.25])
    Arm_Path_Planner.gogogo_sim()
    Arm_Path_Planner.get_now_position()
    time.sleep(1)
    Arm_Path_Planner.set_target([0.5, 0.5, 0.25])
    Arm_Path_Planner.gogogo_sim()
    Arm_Path_Planner.get_now_position()
    time.sleep(1)
    Arm_Path_Planner.set_target([0.6, 0.6, 0.25])
    Arm_Path_Planner.gogogo_sim()
    Arm_Path_Planner.get_now_position()
    time.sleep(1)
    Arm_Path_Planner.set_target([0.25, 0.25, 0.25])
    Arm_Path_Planner.gogogo_sim()
    Arm_Path_Planner.get_now_position()
    time.sleep(1)
    Arm_Path_Planner.set_target([0.21, 0.21, 0.25])
    Arm_Path_Planner.gogogo_sim()
    Arm_Path_Planner.get_now_position()
    time.sleep(1)
    Arm_Path_Planner.set_target([0.2, 0.2, 0.25])
    Arm_Path_Planner.gogogo_sim()
    Arm_Path_Planner.get_now_position()
    time.sleep(1)
    while 1:
        pass
    
if __name__ == "__main__":
    main()