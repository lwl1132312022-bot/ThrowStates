"""
针孔模型工具 —— 像素误差 ↔ 真实距离转换。

针孔相机模型:
    D_real = D_px * Z_C / f

其中:
    D_real  = 真实世界距离 (m)
    D_px    = 图像上的像素距离
    Z_C     = 相机到目标的距离 (m) = altitude - object_height
    f       = 相机焦距 (pixel)

本模块提供 px→m 和 m→px 双向转换函数。

参考: cuadc_ws/src/pinhole_model.py
"""

from dataclasses import dataclass

from config import (
    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
    BUCKET_HEIGHT_M,
)


@dataclass
class PinholeCamera:
    """针孔相机参数。"""

    fx: float = CAMERA_FX    # 焦距 x (pixel)
    fy: float = CAMERA_FY    # 焦距 y (pixel)
    cx: float = CAMERA_CX    # 主点 x (pixel)
    cy: float = CAMERA_CY    # 主点 y (pixel)

    # 目标物体高度（用于修正 Z_C）
    object_height_m: float = BUCKET_HEIGHT_M  # 桶高 30cm


# 默认相机实例（Orin NX 下视, 1280×720）
DEFAULT_CAMERA = PinholeCamera()


def px_to_m(dx_px: float, dy_px: float,
            altitude_m: float,
            cam: PinholeCamera | None = None) -> tuple[float, float]:
    """
    针孔模型: 像素误差 → 真实距离。

    参数:
        dx_px, dy_px: 图像坐标系下的像素偏移
                      dx>0: 目标在图像右侧 → 飞机需右移(东)
                      dy>0: 目标在图像下方 → 飞机需前移(北)
        altitude_m:   当前高度 (m, 相对地面)
        cam:          相机参数

    返回:
        (dx_m, dy_m): 真实世界距离 (m)

    公式: D_real = D_px * (altitude - object_height) / f
    """
    if cam is None:
        cam = DEFAULT_CAMERA

    z_c = altitude_m - cam.object_height_m
    if z_c <= 0.0:
        return 0.0, 0.0  # 高度无效（太低或在地面以下）

    dx_m = dx_px * z_c / cam.fx
    dy_m = dy_px * z_c / cam.fy
    return dx_m, dy_m


def m_to_px(dx_m: float, dy_m: float,
            altitude_m: float,
            cam: PinholeCamera | None = None) -> tuple[float, float]:
    """
    针孔模型逆变换: 真实距离 → 像素偏移。

    用于 SITL 仿真中模拟视觉检测结果。
    """
    if cam is None:
        cam = DEFAULT_CAMERA

    z_c = altitude_m - cam.object_height_m
    if z_c <= 0.0:
        return 0.0, 0.0

    dx_px = dx_m * cam.fx / z_c
    dy_px = dy_m * cam.fy / z_c
    return dx_px, dy_px
