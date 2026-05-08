import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import threading


class Arm_Path_Planner:
    def __init__(self, model_path, site_name="ee_site", num_steps=100, max_steps=100, tol=1e-4):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"在XML模型中找不到名为 '{site_name}' 的 site！请检查你的MJCF。")
        self.target_3d_position = None
        self.max_steps = max_steps
        self.num_steps = num_steps
        self.tol = tol
        self._thread_running = False
        self._physics_thread = None
        self._muoco_lock = threading.Lock()
        self.__is_Descartes=False
    # ==========================================
    # 公开 API
    # ==========================================
    def get_now_position(self):
        with self._muoco_lock:
            mujoco.mj_kinematics(self.model, self.data)
            now_pos = self.data.site_xpos[self.site_id].copy()
        print(f"当前位置: {now_pos}")
        return now_pos

    def get_target_position(self):
        if self.target_3d_position is not None:
            print(f"目标位置: {self.target_3d_position}")
        else:
            print("未设置目标位置")
        return self.target_3d_position
    # 参数输入的格式是[x,y,z],其中范围是
    # x：

    def set_target(self, target_3d_position, is_Descartes=False):
        self.target_3d_position = np.array(target_3d_position)
        self.__is_Descartes=is_Descartes
    def init_sim(self):
        """打开仿真窗口并启动后台物理线程，立即返回（非阻塞）。
           不用 with，否则退出 with 块时 viewer 会被自动 close。"""
        if self.viewer is not None:
            print("仿真已初始化")
            return
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        time.sleep(0.5)
        self._start_physics_thread()

    def gogogo_sim(self):
        """执行运动规划并播放路点（阻塞直到运动完成）"""
        if self.target_3d_position is None:
            print("错误：未设置目标位置，请先调用 set_target()")
            return
        if self.viewer is None:
            print("错误：仿真未初始化，请先调用 init_sim()")
            return

        print("正在生成路点...")
        waypoints = self._generate_waypoints()

        print("开始沿规划路点运动...")
        for waypoint in waypoints:
            if not self.viewer.is_running():
                break
            with self._muoco_lock:
                self.data.ctrl[:] = waypoint
            time.sleep(0.04)  # 给物理线程 ~20 步的时间驱动关节
        print("到达目标位置！")
        time.sleep(1)
    def close(self):
        self._thread_running = False
        if self._physics_thread and self._physics_thread.is_alive():
            self._physics_thread.join(timeout=2.0)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ==========================================
    # 后台物理线程（唯一的 mj_step / sync 调用者）
    # ==========================================
    def _start_physics_thread(self):
        if self._thread_running:
            print("物理线程已在运行")
            return
        self._thread_running = True
        self._physics_thread = threading.Thread(
            target=self._physics_loop,
            daemon=True
        )
        self._physics_thread.start()
        print("后台物理线程已启动")

    def _physics_loop(self):
        """所有 mj_step 和 viewer.sync 都在这一个线程里，避免多线程竞争 MuJoCo 数据"""
        while self._thread_running and self.viewer.is_running():
            with self._muoco_lock:
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.005)

    # ==========================================
    # IK 与路点生成（内部方法，调用时持有锁）
    # ==========================================
    def _compute_ik(self, target_pos):
        q_init = self.data.qpos.copy()

        for _ in range(self.max_steps):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            current_pos = self.data.site_xpos[self.site_id]
            error = target_pos - current_pos

            if np.linalg.norm(error) < self.tol:
                break

            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)

            damping = 1e-3
            J = jacp
            J_T = J.T
            dq = J_T @ np.linalg.inv(J @ J_T + damping**2 * np.eye(3)) @ error

            step_size = 0.5
            self.data.qpos[:] += dq * step_size

        q_target = self.data.qpos.copy()
        self.data.qpos[:] = q_init
        mujoco.mj_kinematics(self.model, self.data)
        return q_target

    def _generate_waypoints(self):
        """生成路点时持有锁，阻止物理线程同时读写 data"""
        with self._muoco_lock:
            mujoco.mj_kinematics(self.model, self.data)
            now = self.data.site_xpos[self.site_id].copy()
            target = self.target_3d_position.copy()

            print(f"当前位置: {now}")
            print(f"目标位置: {target}")

            joint_waypoints = []
            original_qpos = self.data.qpos.copy()
            if self.__is_Descartes:
                for i in range(1, self.num_steps + 1):
                    interp_pos = now + (target - now) * (i / self.num_steps)
                    q_waypoint = self._compute_ik(interp_pos)
                    joint_waypoints.append(q_waypoint)
                    self.data.qpos[:] = q_waypoint

                self.data.qpos[:] = original_qpos
                mujoco.mj_kinematics(self.model, self.data)
            else:
                 # 2. 用一次 IK 求出目标关节角（终点）
                q_target = self._compute_ik(self.target_3d_position)

                # 3. 恢复当前状态（_compute_ik 会内部恢复，但为了清晰可再设一次）
                self.data.qpos[:] = original_qpos
                mujoco.mj_kinematics(self.model, self.data)

                # 4. 在关节空间进行线性插值
                joint_waypoints = []
                for i in range(1, self.num_steps + 1):
                    alpha = i / self.num_steps
                    q_waypoint = original_qpos + (q_target - original_qpos) * alpha
                    joint_waypoints.append(q_waypoint.copy())
        print(f"成功生成了 {len(joint_waypoints)} 个中间关节路点！")
        return joint_waypoints

    def __del__(self):
        self.close()
