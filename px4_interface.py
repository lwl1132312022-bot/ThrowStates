"""
PX4 通信层 —— 简化版 MAVSDK 接口。
====================================
封装 offboard 控制、舵机、遥测查询。

用法:
    from px4_interface import PX4Interface

    interface = PX4Interface()
    await interface.connect_and_setup()
    await interface.arm()
    await interface.takeoff(3.0)
    await interface.switch_to_offboard()
    # ... 视觉循环 ...
    await interface.set_actuator(7, 1.0)  # 释放舵机
    await interface.disarm()

参考: cuadc_ws/src/interface.py
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from mavsdk import System
from mavsdk.offboard import PositionNedYaw

from config import (
    FIELD_YAW_DEG,
    OFFBOARD_HEARTBEAT_HZ,
    TAKEOFF_COMPLETE_THRESHOLD,
    GPS_FIX_MIN,
    SITL_ADDRESS, SERIAL_PORT, SERIAL_BAUD, CONNECTION_MODE,
    CRUISE_ALTITUDE_M,
)


@dataclass
class HealthStatus:
    """飞控健康状态快照。"""

    is_connected: bool = False
    is_armed: bool = False
    is_offboard: bool = False
    is_global_position_ok: bool = False
    is_home_position_ok: bool = False
    battery_pct: float = 100.0
    estimator_flags_ok: bool = True
    gps_fix_type: int = 3
    altitude_m: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return all([
            self.is_connected,
            self.is_armed,
            self.is_global_position_ok,
            self.is_home_position_ok,
            self.estimator_flags_ok,
            self.gps_fix_type >= GPS_FIX_MIN,
        ])

    def health_detail(self) -> str:
        checks = {
            "connected": self.is_connected,
            "armed": self.is_armed,
            "gps_ok": self.is_global_position_ok,
            "home_ok": self.is_home_position_ok,
            "estimator": self.estimator_flags_ok,
            "gps_fix": self.gps_fix_type >= GPS_FIX_MIN,
        }
        parts = [f"{'✓' if ok else '✗'}{name}" for name, ok in checks.items()]
        parts.append(f"fix={self.gps_fix_type}")
        parts.append(f"alt={self.altitude_m:.1f}m")
        return " ".join(parts)


class PX4Interface:
    """简化版 PX4 通信接口。"""

    def __init__(self, connection_mode: str = CONNECTION_MODE):
        self.drone = System()
        self.health = HealthStatus()
        self._connection_mode = connection_mode

        # 心跳状态
        self._last_setpoint = PositionNedYaw(0.0, 0.0, 0.0, 0.0)
        self._heartbeat_running = False

        # EKF→HOME 原点偏移
        self._home_offset_n: float = 0.0
        self._home_offset_e: float = 0.0
        self._home_offset_d: float = 0.0

        # 跨状态共享数据
        self.shared: dict = {}

    # ------------------------------------------------------------------
    # 连接与初始化
    # ------------------------------------------------------------------

    async def connect_and_setup(self):
        """连接到 PX4，等待 GPS 锁定，启动心跳。"""
        if self._connection_mode == "serial":
            address = f"serial://{SERIAL_PORT}:{SERIAL_BAUD}"
        else:
            address = SITL_ADDRESS

        print(f"[接口] 连接 PX4: {address}")
        await self.drone.connect(system_address=address)

        # 等待连接
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break
        self.health.is_connected = True
        print("[接口] 已连接到飞控")

        # 等待 GPS 和家点位置
        print("[接口] 等待 GPS 锁定...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                break
        self.health.is_global_position_ok = True
        self.health.is_home_position_ok = True
        print("[接口] GPS 已锁定，家点位置已记录")

        # 启动心跳
        asyncio.create_task(self._heartbeat_loop())
        print(f"[接口] 心跳已启动 ({OFFBOARD_HEARTBEAT_HZ} Hz)")

        # 设置 PX4 参数
        await self._setup_params()

    async def _setup_params(self):
        """设置 PX4 全局速度参数。"""
        try:
            await self.drone.param.set_param_float("MPC_XY_VEL_MAX", 4.0)
            await self.drone.param.set_param_float("MPC_XY_CRUISE", 3.0)
            print("[接口] PX4 速度参数已设置")
        except Exception as e:
            print(f"[接口] PX4 参数设置失败: {e}")

    # ------------------------------------------------------------------
    # 起飞前的 Arm
    # ------------------------------------------------------------------

    async def arm(self):
        """上锁并记录 EKF→HOME 偏移。"""
        # 记录 EKF 偏移
        async for odom in self.drone.telemetry.odometry():
            self._home_offset_n = odom.position_body.x_m
            self._home_offset_e = odom.position_body.y_m
            self._home_offset_d = odom.position_body.z_m
            break
        print(f"[接口] EKF→HOME 偏移已记录: "
              f"N({self._home_offset_n:.2f}) E({self._home_offset_e:.2f})")

        await self.drone.action.arm()
        self.health.is_armed = True
        print("[接口] 已上锁")

    # ------------------------------------------------------------------
    # 起飞
    # ------------------------------------------------------------------

    async def takeoff(self, altitude_m: float = CRUISE_ALTITUDE_M,
                      timeout_s: float = 60):
        """
        使用 PX4 内建起飞逻辑爬升至目标高度。
        """
        await self.drone.action.set_takeoff_altitude(altitude_m)
        await self.drone.action.takeoff()
        print(f"[起飞] PX4 内建起飞，目标高度 {altitude_m:.1f} 米")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            alt = await self._read_altitude_direct()
            if alt >= altitude_m * (1 - TAKEOFF_COMPLETE_THRESHOLD):
                print(f"[起飞] 已到达巡航高度 {alt:.1f} 米")
                return
            await asyncio.sleep(0.5)

        raise TimeoutError(
            f"起飞超时 ({timeout_s}s)，"
            f"当前高度 {await self._read_altitude_direct():.1f}m,"
            f"目标 {altitude_m:.1f}m")

    # ------------------------------------------------------------------
    # Offboard 模式
    # ------------------------------------------------------------------

    async def switch_to_offboard(self):
        """
        切换到 offboard 模式。
        必须在起飞完成后调用。
        """
        current_pos = await self.get_position_ned()
        self.update_setpoint(current_pos)
        print(f"[接口] 初始 offboard setpoint: "
              f"N({current_pos.north_m:.1f}) E({current_pos.east_m:.1f}) "
              f"D({current_pos.down_m:.1f})")

        await self.drone.offboard.set_position_ned(self._last_setpoint)
        await self.drone.offboard.start()
        self.health.is_offboard = True
        print("[接口] Offboard 模式已启用")

    # ------------------------------------------------------------------
    # 心跳
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self):
        """以固定频率发送 setpoint。"""
        self._heartbeat_running = True
        interval = 1.0 / OFFBOARD_HEARTBEAT_HZ
        while self._heartbeat_running:
            try:
                await self.drone.offboard.set_position_ned(
                    self._last_setpoint)
            except Exception as e:
                print(f"[调试] 心跳 setpoint 发送失败: {e}")
            await asyncio.sleep(interval)
        print("[接口] 心跳循环已停止")

    def stop_heartbeat(self):
        """停止心跳循环。"""
        self._heartbeat_running = False

    # ------------------------------------------------------------------
    # Setpoint 更新
    # ------------------------------------------------------------------

    def update_setpoint(self, setpoint: PositionNedYaw):
        """更新位置 setpoint。"""
        self._last_setpoint = setpoint

    def get_yaw(self) -> float:
        """当前航向角。"""
        return FIELD_YAW_DEG

    # ------------------------------------------------------------------
    # 舵机控制
    # ------------------------------------------------------------------

    async def set_actuator(self, index: int, value: float):
        """通过 AUX 输出控制舵机。"""
        await self.drone.action.set_actuator(index, value)
        print(f"[舵机] AUX{index} -> {value:.2f}")

    # ------------------------------------------------------------------
    # 降落与关停
    # ------------------------------------------------------------------

    async def land(self):
        """指令自动降落。"""
        await self.drone.action.land()
        print("[接口] 降落指令已发送")

    async def disarm(self):
        """规范关停: 停止心跳 → 退出 offboard → 上锁。"""
        self.stop_heartbeat()
        await asyncio.sleep(0.1)

        try:
            await self.drone.offboard.stop()
            print("[接口] 已退出 offboard 模式")
        except Exception as e:
            print(f"[接口] 退出 offboard 失败: {e}")

        try:
            await self.drone.action.disarm()
            self.health.is_armed = False
            self.health.is_offboard = False
            print("[接口] 已断开上锁")
        except Exception as e:
            print(f"[接口] 断开上锁失败: {e}")

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------

    async def global_guard_check(self,
                                 allow_disarmed: bool = False) -> bool:
        """健康检查。节流至约 2 Hz。"""
        now = time.monotonic()
        if not hasattr(self, "_last_guard_read"):
            self._last_guard_read = 0.0
        if not hasattr(self, "_cached_healthy"):
            self._cached_healthy = True

        if now - self._last_guard_read > 0.5:
            self._last_guard_read = now
            try:
                async for state in self.drone.core.connection_state():
                    self.health.is_connected = state.is_connected
                    break
                async for armed in self.drone.telemetry.armed():
                    self.health.is_armed = armed
                    break
                async for health in self.drone.telemetry.health():
                    self.health.is_global_position_ok = \
                        health.is_global_position_ok
                    self.health.is_home_position_ok = \
                        health.is_home_position_ok
                    break
                async for gps in self.drone.telemetry.gps_info():
                    self.health.gps_fix_type = gps.fix_type.value
                    break
                async for pos in self.drone.telemetry.position():
                    self.health.altitude_m = pos.relative_altitude_m
                    break
            except Exception as e:
                print(f"[调试] 健康检查异常: {e}")

            if allow_disarmed:
                self._cached_healthy = (
                    self.health.is_connected
                    and self.health.is_global_position_ok
                    and self.health.is_home_position_ok
                    and self.health.estimator_flags_ok
                    and self.health.gps_fix_type >= GPS_FIX_MIN
                )
            else:
                self._cached_healthy = self.health.is_healthy

        return self._cached_healthy

    # ------------------------------------------------------------------
    # 遥测查询
    # ------------------------------------------------------------------

    async def get_position_ned(self) -> PositionNedYaw:
        """获取当前 NED 位置 (减去 HOME 偏移)。"""
        async for odom in self.drone.telemetry.odometry():
            return PositionNedYaw(
                odom.position_body.x_m - self._home_offset_n,
                odom.position_body.y_m - self._home_offset_e,
                odom.position_body.z_m - self._home_offset_d,
                FIELD_YAW_DEG,
            )

    async def get_altitude(self) -> float:
        """获取当前相对高度 (米)。"""
        async for odom in self.drone.telemetry.odometry():
            return -(odom.position_body.z_m - self._home_offset_d)

    async def _read_altitude_direct(self) -> float:
        """直接读取相对高度 (同 get_altitude)。"""
        async for odom in self.drone.telemetry.odometry():
            return -(odom.position_body.z_m - self._home_offset_d)
