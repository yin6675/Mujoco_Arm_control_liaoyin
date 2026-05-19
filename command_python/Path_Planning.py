import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import threading

# 构造参数解释：
    # model_path: 模型文件路径
    # site_name: 末端执行器在模型中的名称
    # num_steps: 每次运动规划中计算的路点数量
    # max_steps: 最大运动规划步数
    # tol: 运动规划停止的误差阈值
class Arm_Path_Planner:
    
    def __init__(self, model_path, site_name="ee_site", reference_site_name="reference_origin",
                 num_steps=100, max_steps=100, tol=1e-4):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"在XML模型中找不到名为 '{site_name}' 的 site！请检查你的MJCF。")
        self.reference_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, reference_site_name)
        if self.reference_site_id == -1:
            raise ValueError(f"在XML模型中找不到名为 '{reference_site_name}' 的 site！请检查你的MJCF。")
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
            ee_world = self.data.site_xpos[self.site_id].copy()
            ref_world = self.data.site_xpos[self.reference_site_id].copy()
            now_pos = ee_world - ref_world
        print(f"当前位置(相对于参考原点): {now_pos}")
        return now_pos

    def get_target_position(self):
        if self.target_3d_position is not None:
            print(f"目标位置: {self.target_3d_position}")
        else:
            print("未设置目标位置")
        return self.target_3d_position
    # 参数输入的格式是[x,y,z],其中范围是
    # x：0.1 ~ 0.5
    # y：-0.5 ~ 0.5
    # z：0 ~ -0.41
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
            if not self._thread_running or not self.viewer.is_running():
                break
            with self._muoco_lock:
                self.data.ctrl[:] = waypoint
            # 用多次短 sleep 代替一次长 sleep，提高 Ctrl+C 响应速度
            time.sleep(0.04)
        if self._thread_running:
            print("到达目标位置！")
        time.sleep(1)
    def close(self):
        self._thread_running = False
        # 不 join 物理线程 —— 它可能卡在 viewer.sync() 的 native GL 调用里，
        # join 会导致主线程也一起卡死。daemon 线程随进程退出自动清理。
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
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
        while self._thread_running and self.viewer is not None \
              and self.viewer.is_running():
            with self._muoco_lock:
                mujoco.mj_step(self.model, self.data)
            try:
                self.viewer.sync()
            except Exception:
                break
            time.sleep(0.005)

    # ==========================================
    # IK 与路点生成（内部方法，调用时持有锁）
    # ==========================================
    def _compute_ik(self, target_pos):
        q_init = self.data.qpos.copy()

        # 构建关节限位：从 jnt_range 映射到 qpos 索引
        q_min = np.full(self.model.nv, -np.inf)
        q_max = np.full(self.model.nv, np.inf)
        for jnt_id in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt_id]
            if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                dof_addr = self.model.jnt_dofadr[jnt_id]
                q_min[dof_addr] = self.model.jnt_range[jnt_id][0]
                q_max[dof_addr] = self.model.jnt_range[jnt_id][1]

        best_q = q_init.copy()
        best_error = float('inf')

        for _ in range(self.max_steps):
            mujoco.mj_kinematics(self.model, self.data)

            current_pos = self.data.site_xpos[self.site_id]
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)

            if error_norm < best_error:
                best_error = error_norm
                best_q = self.data.qpos.copy()

            if error_norm < self.tol:
                break

            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)

            # 自适应阻尼：基于雅可比最小奇异值，边界附近自动增大阻尼
            s = np.linalg.svd(jacp, compute_uv=False)
            damping = max(s[-1] * 0.1, 1e-2)

            J_T = jacp.T
            dq = J_T @ np.linalg.inv(jacp @ J_T + damping**2 * np.eye(3)) @ error

            # 自适应步长：误差大时减小步长，防止冲过限位
            step_size = min(0.3, 1.0 / (1.0 + error_norm * 10.0))
            dq = dq * step_size

            # 钳位到关节限位
            self.data.qpos[:] = np.clip(self.data.qpos + dq, q_min, q_max)

        # 返回迭代过程中的最优解
        self.data.qpos[:] = q_init
        mujoco.mj_kinematics(self.model, self.data)

        if best_error > 0.01:
            print(f"警告: IK 未完全收敛, 最优误差={best_error:.4f}m")

        return best_q

    def _generate_waypoints(self):
        """生成路点时持有锁，阻止物理线程同时读写 data"""
        with self._muoco_lock:
            mujoco.mj_kinematics(self.model, self.data)
            ref_world = self.data.site_xpos[self.reference_site_id].copy()
            ee_world = self.data.site_xpos[self.site_id].copy()
            now = ee_world - ref_world
            # 用户输入的目标是相对于参考原点的，转换为世界坐标用于 IK
            target_world = self.target_3d_position.copy() + ref_world

            print(f"当前位置(相对参考原点): {now}")
            print(f"目标位置(相对参考原点): {self.target_3d_position}")

            joint_waypoints = []
            original_qpos = self.data.qpos.copy()
            if self.__is_Descartes:
                for i in range(1, self.num_steps + 1):
                    interp_pos = ee_world + (target_world - ee_world) * (i / self.num_steps)
                    q_waypoint = self._compute_ik(interp_pos)
                    joint_waypoints.append(q_waypoint)
                    self.data.qpos[:] = q_waypoint

                self.data.qpos[:] = original_qpos
                mujoco.mj_kinematics(self.model, self.data)
            else:
                 # 2. 用一次 IK 求出目标关节角（终点）
                q_target = self._compute_ik(target_world)

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
        #调用打印过程中各个路点的函数
        # self._print_waypoint_summary(joint_waypoints, ref_world)
        return joint_waypoints

    def _joint_to_ee_pos(self, q):
        """将一组关节角度转为末端相对坐标"""
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        ee_world = self.data.site_xpos[self.site_id].copy()
        ref_world = self.data.site_xpos[self.reference_site_id].copy()
        return ee_world - ref_world

    def _print_waypoint_summary(self, waypoints, ref_world):
        """打印路点的末端三维坐标摘要（首/中/尾）"""
        indices = [0, len(waypoints) // 2, len(waypoints) - 1]
        print("\n===== 路点末端坐标摘要 =====")
        for idx in range(len(waypoints)):
            pos = self._joint_to_ee_pos(waypoints[idx])
            print(f"  路点{idx + 1}/{len(waypoints)}: X={pos[0]:.4f} Y={pos[1]:.4f} Z={pos[2]:.4f}")

    def __del__(self):
        self.close()
