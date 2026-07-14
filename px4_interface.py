"""
PX4 通信层 — 仅连接 + 舵机控制。
===============================
不上锁，AUX 独立供电直接驱动舵机。
室内测试不依赖 GPS，不依赖遥测。

用法:
    interface = PX4Interface()
    await interface.connect_and_setup()
    await interface.set_actuator(7, 1.0)
"""

from mavsdk import System

from config import (
    SITL_ADDRESS, SERIAL_PORT, SERIAL_BAUD, CONNECTION_MODE,
    REQUIRE_GPS,
)


class PX4Interface:
    """最简 PX4 通信 — 连接 + 舵机。"""

    def __init__(self, connection_mode: str = CONNECTION_MODE):
        self.drone = System()
        self._connection_mode = connection_mode

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------

    async def connect_and_setup(self):
        """连接到 PX4。室内模式下不等待 GPS。"""
        if self._connection_mode == "serial":
            address = f"serial://{SERIAL_PORT}:{SERIAL_BAUD}"
        else:
            address = SITL_ADDRESS

        print(f"[PX4] 连接模式: {self._connection_mode}")
        print(f"[PX4] 地址: {address}")
        print(f"[PX4] GPS: {'需要' if REQUIRE_GPS else '跳过 (室内模式)'}")
        await self.drone.connect(system_address=address)

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break
        print("[PX4] MAVSDK 连接已建立")
        print("[PX4] 不上锁 (AUX 独立供电), 可直接控制舵机")

        if REQUIRE_GPS:
            print("[PX4] 等待 GPS 锁定...")
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    break
            print("[PX4] GPS 已锁定 (global+home OK)")
        else:
            print("[PX4] 室内模式 — 不等待 GPS, 直接继续")

    # ------------------------------------------------------------------
    # 舵机
    # ------------------------------------------------------------------

    async def set_actuator(self, index: int, value: float):
        """AUX 输出控制舵机 (不上锁也能用，前提是 AUX 排针有独立供电)。

        Args:
            index: AUX 通道号 (1-16)
            value: -1.0 ~ 1.0
        """
        await self.drone.action.set_actuator(index, value)
        print(f"[舵机] AUX{index} -> {value:.2f}")
