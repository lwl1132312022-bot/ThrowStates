"""
PX4 通信层 — 最简版，仅用于舵机控制。
=====================================

用法:
    interface = PX4Interface()
    await interface.connect_and_setup()
    await interface.arm()
    # ... 视觉检测 + 舵机释放 ...
    await interface.set_actuator(7, 1.0)
    await interface.disarm()
"""

import asyncio
import math

from mavsdk import System

from config import (
    SITL_ADDRESS, SERIAL_PORT, SERIAL_BAUD, CONNECTION_MODE,
    DROP_MAX_ANGULAR_RATE, DROP_MAX_VELOCITY,
)


class PX4Interface:
    """最简 PX4 通信 — 仅连接/上锁/舵机/上锁。"""

    def __init__(self, connection_mode: str = CONNECTION_MODE):
        self.drone = System()
        self._connection_mode = connection_mode

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------

    async def connect_and_setup(self):
        """连接到 PX4，等待 GPS 就绪。"""
        if self._connection_mode == "serial":
            address = f"serial://{SERIAL_PORT}:{SERIAL_BAUD}"
        else:
            address = SITL_ADDRESS

        print(f"[PX4] 连接: {address}")
        await self.drone.connect(system_address=address)

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break
        print("[PX4] 已连接")

        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                break
        print("[PX4] GPS 已锁定")

    # ------------------------------------------------------------------
    # 上锁 / 上锁
    # ------------------------------------------------------------------

    async def arm(self):
        """上锁（舵机供电需要飞控处于 armed 状态）。"""
        await self.drone.action.arm()
        print("[PX4] 已上锁")

    async def disarm(self):
        """上锁。"""
        await self.drone.action.disarm()
        print("[PX4] 已上锁")

    # ------------------------------------------------------------------
    # 舵机
    # ------------------------------------------------------------------

    async def set_actuator(self, index: int, value: float):
        """AUX 输出控制舵机。

        Args:
            index: AUX 通道号 (1-16)
            value: -1.0 ~ 1.0
        """
        await self.drone.action.set_actuator(index, value)
        print(f"[舵机] AUX{index} -> {value:.2f}")

    # ------------------------------------------------------------------
    # 稳定性遥测
    # ------------------------------------------------------------------

    async def get_angular_rate(self) -> float:
        """获取机体角速率模长 (rad/s), 用于判断飞机是否稳定。"""
        try:
            async for rate in self.drone.telemetry.attitude_angular_velocity_body():
                return math.hypot(rate.roll_rad_s, rate.pitch_rad_s, rate.yaw_rad_s)
        except Exception:
            return 999.0  # 读取失败返回大值, 阻止投放

    async def get_velocity(self) -> float:
        """获取水平速度模长 (m/s), 用于判断飞机是否悬停。"""
        try:
            async for vel in self.drone.telemetry.velocity_ned():
                return math.hypot(vel.north_m_s, vel.east_m_s)
        except Exception:
            return 999.0

    async def is_stable(self) -> tuple[bool, float, float]:
        """检查飞机是否稳定 (角速率≈0, 速度≈0)。

        Returns:
            (stable, angular_rate_mag, velocity_mag)
        """
        ang = await self.get_angular_rate()
        vel = await self.get_velocity()
        stable = ang < DROP_MAX_ANGULAR_RATE and vel < DROP_MAX_VELOCITY
        return stable, ang, vel
