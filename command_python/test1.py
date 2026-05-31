import numpy as np
import time
import signal
import Path_Planning
from HitbotInterface import HitbotInterface
from pynput import keyboard   # 需要安装：pip install pynput

# ========== 全局停止标志和键盘监听 ==========
stop_program = False
# 
def on_press(key):
    global stop_program
    try:
        if key == keyboard.Key.esc:
            print("\n[紧急停止] ESC 键被按下，正在停止机械臂...")
            stop_program = True
            # 如果 robot 对象存在，调用 stop_move()
            if 'robot' in globals() and robot is not None:
                robot.stop_move()
                print("[已执行] robot.stop_move()，机械臂将平滑停止。")
            return False  # 停止监听器
    except Exception as e:
        print(f"键盘回调错误: {e}")

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener

# ========== 以下是你的原有代码（保留所有功能） ==========
def init_robot(robot_id, generation, z_travel):
    
    # 实例化机械臂对象
    robot = HitbotInterface(robot_id)
    # 1.初始化网络端口，使用其他API前必须先调用此函数进行初始化，否则造成程序崩溃
    net_ret = robot.net_port_initial()
    if net_ret != 1:
        print(f"网络端口初始化失败，错误码：{net_ret}")
        return None
    
    # 初始化完成后，进行连接机械臂
    # 2.等待机械臂连接，True=0表示机械臂在线 False=1表示机械臂不在线
    ret = robot.is_connect()
    while ret != 1:
        time.sleep(0.1)
        ret = robot.is_connect()
    print("机械臂连接成功")

    # 3.初始化机械臂参数
    init_ret = robot.initial(generation, z_travel)
    if init_ret == 1:
        print("机械臂初始化成功！")

        # 4.机械臂初始化成功后，解锁机械臂，使其可接收运动指令
        robot.unlock_position()
        return robot   # 5.返回机械臂对象供后续使用
    else:
        error_info = {
            0: "机械臂不在线",
            2: "generation参数错误",
            3: "机械臂当前位置不在限定范围，需用joint_home强制回零",
            12: "z_travel传参错误",
            101: "传入参数非数字",
            105: "存在关节失效",
            10000: "pid自检异常"
        }
        print(f"robot initial failed，错误码：{init_ret}，原因：{error_info.get(init_ret, '未知错误')}")#若机械臂初始化失败，打印错误码及原因
        return None

    # 6.运动前先进行检测机械臂1-4轴关节状态，正常返回True，异常返回False
def check_joint_state(robot: HitbotInterface):

    print("\n开始检测关节状态...")
    #遍历1-4轴关节
    for joint_num in range(1, 5):
        state = robot.get_joint_state(joint_num)
        if state == 1:
            print(f"关节{joint_num}状态正常")
        else:
            print(f"关节{joint_num}异常，错误码：{state}")
            return False
    print("所有关节状态正常")
    return True

# 7.切换手系，切换成功返回True，失败返回False
def switch_attitude(robot: HitbotInterface, speed_hand):
    print(f"\n开始切换手系，切换速度：{speed_hand}mm/s")
    switch_ret = robot.change_attitude(speed_hand)
    
    # 解析返回值
    error_info = {
        0: "机械臂正在执行其他指令，本次指令无效",
        1: "手系切换指令生效，开始切换",
        3: "未初始化",
        4: "过程点无法到达",
        6: "伺服未开启",
        7: "过程点无法到达",
        11: "手机端在控制",
        101: "传入参数非数字",
        102: "发生碰撞，本次指令无效",
        103: "轴发生复位，需要重新初始化"
    }
    
    if switch_ret == 1:
        # 等待切换完成
        robot.wait_stop()
        time.sleep(1)
        print("手系切换完成！")
        return True
    else:
        print(f"手系切换失败，错误码：{switch_ret}，原因：{error_info.get(switch_ret, '未知错误')}")
        return False

def main(robot: HitbotInterface):
    global stop_program
    planner = Path_Planning.Arm_Path_Planner(
        model_path="../my_robot_description/urdf/Arm_mujoco.xml",
        site_name="ee_site",
        reference_site_name="reference_origin",
        num_steps=200, max_steps=200, tol=1e-3
    )
    interrupted = False

    # 启动键盘监听（ESC键停止）
    listener = start_keyboard_listener()
    print("[安全系统] 已启动键盘监听，按下 ESC 键可立即停止机械臂。")

    # 如果使用真实机械臂，则动态替换 planner.gogogo_real 为带停止检查的版本
    if robot is not None:
        def safe_gogogo_real(robot, speed=200, roughly=1):
            """带停止标志的版本，使用关节角度模式"""
            if stop_program:
                print("已收到停止信号，取消运动。")
                return
            if planner.target_3d_position is None:
                print("错误：未设置目标位置")
                return
            waypoints = planner._generate_waypoints()
            if not waypoints:
                return
            print(f"开始沿 {len(waypoints)} 个路点运动（关节角度模式, roughly={roughly})")
            for i, q_waypoint in enumerate(waypoints):
                if stop_program:
                    print("检测到停止信号，终止后续路点发送。")
                    break
                # 正确索引：q[0]=z, q[1]=angle1, q[2]=angle2, q[3]=r
                z_m = q_waypoint[0]
                angle1_rad = q_waypoint[1]
                angle2_rad = q_waypoint[2]
                r_rad = q_waypoint[3]

                angle1_deg = np.degrees(angle1_rad)
                angle2_deg = np.degrees(angle2_rad)
                z_mm = -z_m * 1000.0   # 取反
                r_deg = np.degrees(r_rad)

                print(f"路点 {i+1}/{len(waypoints)}: "
                      f"angle1={angle1_deg:.2f}°, angle2={angle2_deg:.2f}°, "
                      f"Z={z_mm:.2f}mm, R={r_deg:.2f}°")
                move_ret = robot.new_movej_angle(angle1_deg, angle2_deg, z_mm, r_deg, speed, roughly)
                if move_ret != 1:
                    print(f"运动指令失败，错误码 {move_ret}，终止运动")
                    break
                time.sleep(0.05)
            if not stop_program:
                robot.wait_stop()
                print("真实机械臂运动完成")
            else:
                print("运动已由用户停止。")
        planner.gogogo_real = safe_gogogo_real  # 替换方法

    # 纯仿真模式：robot 为 None 时跳过所有真实机械臂操作
    if robot is not None:
        # 1. 检测真实机械臂关节状态
        if not check_joint_state(robot):
            print("关节状态异常，无法运动")
            return
        # 2. 记录初始位置（仅显示）
        robot.get_scara_param()
        init_x, init_y, init_z, init_r = robot.x, robot.y, robot.z, robot.r
        print(f"\n初始位置：X={init_x}mm, Y={init_y}mm, Z={init_z}mm, R={init_r}deg")  
    else:
        print("纯仿真模式：不连接真实机械臂。")

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n收到中断信号，正在退出...")
    signal.signal(signal.SIGINT, on_interrupt)

    try:
        # 初始化仿真窗口（即使只运行 real，也需要调用 init_sim() 来生成路点？）
        # 注意：gogogo_real 需要调用 _generate_waypoints，而它需要 self.viewer 可能不需要，
        # 但为了安全，还是初始化仿真（可以不显示窗口？若不需要画面，可注释掉 init_sim）
        # 如果不想要仿真窗口，可以注释掉 init_sim，但需确保 model 已加载，路点生成可用。
        planner.init_sim()   # 如果你完全不想要仿真画面，可以注释掉这行
        planner.get_now_position()

        # ========== 测试点1 ==========
        planner.set_target([0.3, -0.3, -0.25])

        # # 选项A：先仿真测试（注释掉这行，看轨迹）
        if not interrupted:
            planner.gogogo_sim()

        # 选项B：确认轨迹没问题后，改为真实运动（取消下面的注释，并注释上面的仿真）
        # if not interrupted:
        #     planner.gogogo_real(robot, speed=150, roughly=1)

        # ========== 测试点2 ==========
        # planner.set_target([0.4, -0.4, -0.25])

        # 选项A：先仿真测试（注释掉这行，看轨迹）
        # if not interrupted:
        #     planner.gogogo_sim()

        # 选项B：确认轨迹没问题后，改为真实运动（取消下面的注释，并注释上面的仿真）
        # if not interrupted:
        #     planner.gogogo_real(robot, speed=150, roughly=1)

        # 保持窗口，等待用户关闭（如果你不需要仿真画面，可以简化）
        while not interrupted and planner.viewer is not None and planner.viewer.is_running():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nCtrl+C 收到，正在退出...")
    finally:
        planner.close()
        print("资源已清理，进程退出。")

# 真实机械臂模式（默认启用）
# if __name__ == "__main__":
#     # 机械臂ID
#     ROBOT_ID = 175
#     # 400mm臂展传1，320mm及其他传5
#     ROBOT_GENERATION = 5
#     # 上下关节有效行程，单位mm
#     ROBOT_Z_TRAVEL = 410
#     # 执行初始化,
#     robot = init_robot(ROBOT_ID, ROBOT_GENERATION, ROBOT_Z_TRAVEL)
#     # 后续可通过robot对象调用各类控制接口实现机械臂操控
#     if robot:
#         main(robot)

# 纯仿真模式（如需使用，请注释上面的块，取消下面块的注释）
if __name__ == "__main__":
    # 纯仿真测试（不连接真实机械臂）
    robot = None
    main(robot)
