import numpy as np
import time
import signal
import Path_Planning


def main():
    planner = Path_Planning.Arm_Path_Planner(
        model_path="../my_robot_description/urdf/Arm_mujoco.xml",
        site_name="ee_site",
        reference_site_name="reference_origin",
        num_steps=200, max_steps=200, tol=1e-3
    )

    interrupted = False

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n收到中断信号，正在退出...")
    signal.signal(signal.SIGINT, on_interrupt)

    try:
        planner.init_sim()

        # 测试1
        # planner.get_now_position()
        planner.set_target([0.3, -0.3, -0.25])
        if not interrupted:
            planner.gogogo_sim()
        # planner.get_now_position()
        # # 测试2
        # planner.set_target([0.3, 0.3, -0.25])
        # if not interrupted:
        #     planner.gogogo_sim()
        # planner.get_now_position()
        # # 保持窗口，响应 Ctrl+C 或等待用户关闭
        # while not interrupted and planner.viewer is not None \
        #       and planner.viewer.is_running():
        #     time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nCtrl+C 收到，正在退出...")
    finally:
        planner.close()
        print("资源已清理，进程退出。")


if __name__ == "__main__":
    main()
