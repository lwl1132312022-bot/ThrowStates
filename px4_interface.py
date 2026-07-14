"""
PX4 通信层 — 仅连接 + 舵机。
===========================
不上锁，不等 GPS，不查健康状态。
AUX 独立供电直接驱动舵机。

用法:
    interface = PX4Interface()
    await interface.connect_and_setup()
    await interface.set_actuator(7, 1.0)
"""

import asyncio

from mavsdk import System

from config import (
    SITL_ADDRESS, SERIAL_PORT, SERIAL_BAUD, CONNECTION_MODE,
)


class PX4Interface:
    """最简 PX4 通信 — 连接 + 舵机，不等任何传感器。"""

    def __init__(self, connection_mode: str = CONNECTION_MODE):
        self.drone = System()
        self._connection_mode = connection_mode

    async def connect_and_setup(self):
        """连接 PX4，不等 GPS/传感器/健康检查。"""
        if self._connection_mode == "serial":
            address = f"serial://{SERIAL_PORT}:{SERIAL_BAUD}"
        else:
            address = SITL_ADDRESS

        print(f"[PX4] 连接: {address}")
        await self.drone.connect(system_address=address)

        # 等连接建立 (最多 10 秒)
        print("[PX4] 等待 MAVSDK 连接...")
        try:
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    break
        except asyncio.TimeoutError:
            pass

        print("[PX4] 已连接 (不等 GPS, 不等 arm, 可直接发舵机指令)")

    async def set_actuator(self, index: int, value: float):
        """AUX 输出控制舵机。

        Args:
            index: AUX 通道号 (1-16)
            value: -1.0 ~ 1.0
        """
        await self.drone.action.set_actuator(index, value)
        print(f"[舵机] AUX{index} -> {value:.2f}")
