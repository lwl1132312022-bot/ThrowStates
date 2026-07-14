#!/usr/bin/env python3
"""
圆检测模块 — Canny 边缘 + Contour + fitEllipse + 针孔相机模型
==============================================================
从 YOLO 检测框 ROI 中提取圆形边缘、检测圆心与半径、计算真实直径。

用法:
    from vision.circle_detector import CircleDetector

    cd = CircleDetector()
    result = cd.detect(frame, bbox)        # → CircleResult | None
    diameter_m = cd.compute_diameter(radius_px, alt_rel_m)

参考:
    cuadc_ws/src/vision/circle_detector.py — 已验证的 Contour+fitEllipse 管道
"""

import cv2
import numpy as np
from typing import Optional, Tuple, NamedTuple

from config import (
    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
    CAMERA_DIST_COEFFS,
    BUCKET_HEIGHT_M, CAMERA_OFFSET_M,
    CIRCLE_GAUSSIAN_BLUR_KSIZE,
    CIRCLE_CANNY_LOW, CIRCLE_CANNY_HIGH, CIRCLE_CANNY_APERTURE,
    CIRCLE_MORPH_CLOSE_KSIZE,
    CIRCLE_CIRCULARITY_THRESHOLD,
    CIRCLE_ROI_PADDING_RATIO, CIRCLE_EDGE_MARGIN_PX,
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


class CircleResult(NamedTuple):
    """单次圆检测的结果"""

    cx_px: int                          # 圆心 x (全帧像素坐标)
    cy_px: int                          # 圆心 y (全帧像素坐标)
    radius_px: int                      # 像素半径
    diameter_px: float                  # 像素直径 (= 2 × radius_px)
    diameter_m: float                   # 估算真实直径 (米), -1.0 表示无法计算
    bbox: Tuple[int, int, int, int]     # 使用的检测框 (x1, y1, x2, y2)
    roi_offset: Tuple[int, int]         # ROI 在全帧中的偏移 (roi_x1, roi_y1)


# ---------------------------------------------------------------------------
# 默认相机内参 — Orin NX 下视摄像头, 1280×720
# 棋盘格标定, 平均重投影误差 0.1784 px
# ---------------------------------------------------------------------------

DEFAULT_CAMERA_MATRIX = np.array(
    [
        [CAMERA_FX, 0.0000, CAMERA_CX],
        [0.0000, CAMERA_FY, CAMERA_CY],
        [0.0000, 0.0000, 1.0000],
    ],
    dtype=np.float64,
)

DEFAULT_DIST_COEFFS = np.array(CAMERA_DIST_COEFFS, dtype=np.float64)


# ---------------------------------------------------------------------------
# 圆检测器
# ---------------------------------------------------------------------------


class CircleDetector:
    """在 YOLO 检测框 ROI 内进行 Canny+Contour+fitEllipse 圆检测。

    使用 Contour+fitEllipse 而非 HoughCircles:
      - 桶口在倾斜视角下呈椭圆, fitEllipse 比 HoughCircles 更鲁棒
      - convexHull 闭合 Canny 弧段断口, 使弧段近似圆 → 圆度大幅提升
      - 圆度过滤剔除不规则噪声轮廓
    """

    def __init__(
        self,
        camera_matrix: np.ndarray = DEFAULT_CAMERA_MATRIX,
        dist_coeffs: np.ndarray = DEFAULT_DIST_COEFFS,
        bucket_height_m: float = BUCKET_HEIGHT_M,
        camera_offset_m: float = CAMERA_OFFSET_M,
        # ROI 参数
        roi_padding_ratio: float = CIRCLE_ROI_PADDING_RATIO,
        edge_margin_px: int = CIRCLE_EDGE_MARGIN_PX,
        # Canny 参数
        gaussian_blur_ksize: Tuple[int, int] = CIRCLE_GAUSSIAN_BLUR_KSIZE,
        canny_low: int = CIRCLE_CANNY_LOW,
        canny_high: int = CIRCLE_CANNY_HIGH,
        canny_aperture: int = CIRCLE_CANNY_APERTURE,
        # 形态学闭运算
        morph_close_ksize: Tuple[int, int] = CIRCLE_MORPH_CLOSE_KSIZE,
        # Contour+fitEllipse 参数
        circularity_threshold: float = CIRCLE_CIRCULARITY_THRESHOLD,
        use_convex_hull: bool = True,
        # 可视化
        edge_color: Tuple[int, int, int] = (0, 255, 0),
        circle_color: Tuple[int, int, int] = (0, 0, 255),
        center_color: Tuple[int, int, int] = (0, 0, 255),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        overlay_alpha: float = 0.5,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.fx = float(camera_matrix[0, 0])
        self.fy = float(camera_matrix[1, 1])
        self.cx = float(camera_matrix[0, 2])
        self.cy = float(camera_matrix[1, 2])
        self.bucket_height_m = bucket_height_m
        self.camera_offset_m = camera_offset_m

        self.roi_padding_ratio = roi_padding_ratio
        self.edge_margin_px = edge_margin_px

        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.canny_aperture = canny_aperture

        self.morph_close_ksize = morph_close_ksize
        self.circularity_threshold = circularity_threshold
        self.use_convex_hull = use_convex_hull

        # 可视化参数
        self.edge_color = edge_color
        self.circle_color = circle_color
        self.center_color = center_color
        self.text_color = text_color
        self.overlay_alpha = overlay_alpha

    # ------------------------------------------------------------------
    # 预计算畸变校正映射 (可选)
    # ------------------------------------------------------------------

    def init_undistort_maps(self, width: int, height: int):
        """预计算畸变校正的 remap 映射表 (只需调用一次)。"""
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )
        return map1, map2

    def undistort(self, frame: np.ndarray, map1, map2) -> np.ndarray:
        """对单帧做畸变校正"""
        return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    # ------------------------------------------------------------------
    # 核心检测
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[CircleResult]:
        """
        在 YOLO 检测框 ROI 内检测圆 (Canny 边缘 + Contour+fitEllipse)。

        Args:
            frame: 全帧图像 (BGR, H×W×3)
            bbox:  边界框 (x1, y1, x2, y2) — 像素坐标

        Returns:
            CircleResult — 检测到的圆的信息 (全帧坐标)
            None — 未检测到圆
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1

        # ---- 边缘距检查: 跳过贴近画面边缘的截断框 ----
        if (x1 < self.edge_margin_px or y1 < self.edge_margin_px or
                x2 > w - self.edge_margin_px or y2 > h - self.edge_margin_px):
            print(f"  [圆检测] bbox({x1},{y1},{x2},{y2}) 贴边, 跳过")
            return None

        # 1. 提取 ROI (带 padding)
        roi, roi_x1, roi_y1 = self._extract_roi(frame, x1, y1, x2, y2)
        if roi.size == 0:
            print(f"  [圆检测] ROI 为空, 跳过")
            return None

        # 2. 圆检测
        circle_roi = self._detect_circle_in_roi(roi)
        if circle_roi is None:
            print(f"  [圆检测] bbox({x1},{y1},{x2},{y2}) "
                  f"尺寸={bw}×{bh} → 未检测到圆")
            return None

        cx_roi, cy_roi, radius = circle_roi

        # 3. 转换到全帧坐标
        cx_full = roi_x1 + cx_roi
        cy_full = roi_y1 + cy_roi

        print(f"  [圆检测] bbox({x1},{y1},{x2},{y2}) "
              f"尺寸={bw}×{bh} → 圆@({cx_full},{cy_full}) "
              f"半径={radius}px")

        return CircleResult(
            cx_px=cx_full,
            cy_px=cy_full,
            radius_px=radius,
            diameter_px=2.0 * radius,
            diameter_m=-1.0,  # 调用方用 compute_diameter() 填充
            bbox=(x1, y1, x2, y2),
            roi_offset=(roi_x1, roi_y1),
        )

    def detect_full_frame(self, frame: np.ndarray) -> Optional[CircleResult]:
        """
        在全帧图像上检测圆 (不依赖 YOLO bbox)。
        用作没有 YOLO 模型时的回退方案。

        Args:
            frame: 全帧图像 (BGR, H×W×3)

        Returns:
            CircleResult | None
        """
        h, w = frame.shape[:2]
        # 使用全帧作为 ROI
        bbox = (0, 0, w, h)
        return self.detect(frame, bbox)

    def _extract_roi(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[np.ndarray, int, int]:
        """从帧中提取 ROI (带 padding). 返回 (roi, roi_x1, roi_y1)."""
        h, w = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad_w = int(bw * self.roi_padding_ratio)
        pad_h = int(bh * self.roi_padding_ratio)

        rx1 = max(0, x1 - pad_w)
        ry1 = max(0, y1 - pad_h)
        rx2 = min(w, x2 + pad_w)
        ry2 = min(h, y2 + pad_h)

        if rx2 <= rx1 or ry2 <= ry1:
            rx1, ry1, rx2, ry2 = x1, y1, x2, y2

        roi = frame[ry1:ry2, rx1:rx2]
        return roi, rx1, ry1

    def _detect_circle_in_roi(
        self, roi: np.ndarray
    ) -> Optional[Tuple[int, int, int]]:
        """
        在 ROI 中检测桶口圆 (Contour + fitEllipse)。

        流程:
          灰度化 → 高斯模糊 → Canny → 闭运算 →
          findContours → 最大轮廓 → convexHull → 圆度过滤 → fitEllipse

        Returns:
            (cx_roi, cy_roi, radius) — ROI 内的坐标, 或 None
        """
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return None

        # Step 1: 灰度化 + 高斯模糊
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        blurred = cv2.GaussianBlur(gray, self.gaussian_blur_ksize, 0)

        # Step 2: Canny 边缘检测
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high,
                          apertureSize=self.canny_aperture)

        # Step 3: 形态学闭运算（连接断裂边缘）
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.morph_close_ksize)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Step 4: 查找最外层轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f"    [Canny] 边缘={np.count_nonzero(edges)}px, 轮廓=0 → 无圆")
            return None

        # Step 5: 筛选最大轮廓
        cnt = max(contours, key=cv2.contourArea)

        # Step 6: 凸包 — 闭合 Canny 断口, 让弧段变成近似圆
        hull = cv2.convexHull(cnt) if self.use_convex_hull else cnt
        if len(hull) < 5:
            print(f"    [凸包] 点数={len(hull)} < 5 → 不足以拟合")
            return None

        # Step 7: 圆度过滤 (在凸包上计算, 闭合的弧段 ≈ 圆)
        area = cv2.contourArea(hull)
        peri = cv2.arcLength(hull, True)
        if peri == 0:
            return None
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < self.circularity_threshold:
            print(f"    [圆度] {circularity:.3f} < {self.circularity_threshold} → 剔除")
            return None

        # Step 8: 椭圆拟合 (用原始轮廓, 凸包的弦会扭曲椭圆)
        if len(cnt) < 5:
            return None
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (d1, d2), _angle = ellipse
        diameter_px = (d1 + d2) / 2.0
        radius = int(round(diameter_px / 2.0))

        print(f"    [拟合] 轮廓数={len(contours)} 圆度={circularity:.3f} "
              f"椭圆=({cx:.0f},{cy:.0f}) "
              f"d1={d1:.0f} d2={d2:.0f} 半径={radius}px")

        return int(round(cx)), int(round(cy)), radius

    # ------------------------------------------------------------------
    # 直径计算 (针孔相机模型)
    # ------------------------------------------------------------------

    def compute_diameter(self, radius_px: int, alt_rel_m: float) -> float:
        """
        基于针孔相机模型计算圆的真实直径。

        公式: D_real = (2 × r_px × Z_C) / fx

        其中:
          Z_C = alt_rel_m + camera_offset_m - bucket_height_m
          r_px = 像素半径
          fx   = 焦距 (像素)

        Args:
            radius_px:  圆的像素半径
            alt_rel_m:  相对起飞点高度 (米)

        Returns:
            真实直径 (米), 如果高度无效则返回 -1.0
        """
        z_c = alt_rel_m + self.camera_offset_m - self.bucket_height_m
        if z_c <= 0.0:
            return -1.0
        diameter_px = 2.0 * radius_px
        return (diameter_px * z_c) / self.fx

    # ------------------------------------------------------------------
    # Canny 边缘 (用于可视化, 与检测使用相同参数)
    # ------------------------------------------------------------------

    def get_edges(self, roi: np.ndarray) -> np.ndarray:
        """
        对 ROI 执行与检测一致的 Canny 边缘检测 + 形态学闭运算,
        返回二值边缘图。
        """
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        blurred = cv2.GaussianBlur(gray, self.gaussian_blur_ksize, 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high,
                          apertureSize=self.canny_aperture)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.morph_close_ksize)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges

    # ------------------------------------------------------------------
    # 可视化叠加
    # ------------------------------------------------------------------

    def overlay(
        self,
        frame: np.ndarray,
        roi_x1: int,
        roi_y1: int,
        edges: np.ndarray,
        circle_result: Optional[CircleResult],
    ) -> np.ndarray:
        """
        在帧上叠加 Canny 边缘 (半透明绿色) 和检测圆/圆心 (红色)。

        Args:
            frame:         原始帧 (BGR)
            roi_x1, roi_y1: ROI 在全帧中的左上角坐标
            edges:         Canny 边缘二值图 (与 ROI 同尺寸)
            circle_result: 检测结果 (含坐标和直径信息), 或 None

        Returns:
            叠加后的帧 (新副本)
        """
        result = frame.copy()
        h_roi, w_roi = edges.shape[:2]
        roi_y2 = min(roi_y1 + h_roi, result.shape[0])
        roi_x2 = min(roi_x1 + w_roi, result.shape[1])
        h_roi = roi_y2 - roi_y1
        w_roi = roi_x2 - roi_x1

        if h_roi <= 0 or w_roi <= 0:
            return result

        # Canny 边缘叠加
        edges_cropped = edges[:h_roi, :w_roi]
        roi_region = result[roi_y1:roi_y2, roi_x1:roi_x2]
        green_overlay = np.zeros_like(roi_region)
        green_overlay[edges_cropped > 0] = self.edge_color
        blended = cv2.addWeighted(
            roi_region, 1.0 - self.overlay_alpha,
            green_overlay, self.overlay_alpha, 0,
        )
        result[roi_y1:roi_y2, roi_x1:roi_x2] = blended

        # 圆/圆心/标注
        if circle_result is not None:
            cx = circle_result.cx_px
            cy = circle_result.cy_px
            radius = circle_result.radius_px

            cx = int(np.clip(cx, 0, result.shape[1] - 1))
            cy = int(np.clip(cy, 0, result.shape[0] - 1))

            cv2.circle(result, (cx, cy), radius, self.circle_color, 2)
            cv2.circle(result, (cx, cy), 3, self.center_color, -1)

            cross = max(radius // 2, 3) if radius > 6 else 3
            cv2.line(
                result, (cx - cross, cy), (cx + cross, cy),
                self.center_color, 1,
            )
            cv2.line(
                result, (cx, cy - cross), (cx, cy + cross),
                self.center_color, 1,
            )

            if circle_result.diameter_m > 0:
                label = f"D={circle_result.diameter_m * 100:.1f}cm"
            else:
                label = f"r={radius}px"
            cv2.putText(
                result, label, (cx + radius + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.text_color, 1,
            )

        return result
