#!/usr/bin/env python3
"""
视觉流水线 — YOLO 推理 + 圆检测 + 针孔模型直径计算
=====================================================
将 YOLODetector 与 CircleDetector 串联，提供统一接口供实时飞行使用。

用法:
    from vision import VisionPipeline

    pl = VisionPipeline()
    results = pl.process_frame(frame, alt_rel_m=7.0)

参考: cuadc_ws/src/vision/pipeline.py
"""

from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

from config import (
    CIRCLE_CONF_THRESHOLD,
    YOLO_CONFIDENCE_THRESHOLD, YOLO_IMGSZ,
    BUCKET_HEIGHT_M, CAMERA_OFFSET_M,
    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
    WATER_BOTTLE_OFFSET_M, DROP_ZONE_DIAMETER_M,
)

from .yolo_detector import YOLODetector
from .circle_detector import CircleDetector, CircleResult, DEFAULT_CAMERA_MATRIX, DEFAULT_DIST_COEFFS


class VisionPipeline:
    """YOLO 检测 → 圆检测 → 真实直径 一体的视觉处理流水线"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        camera_matrix: np.ndarray = DEFAULT_CAMERA_MATRIX,
        dist_coeffs: np.ndarray = DEFAULT_DIST_COEFFS,
        bucket_height_m: float = BUCKET_HEIGHT_M,
        camera_offset_m: float = CAMERA_OFFSET_M,
        yolo_conf: float = YOLO_CONFIDENCE_THRESHOLD,
        yolo_iou: float = 0.45,
        imgsz: int = YOLO_IMGSZ,
        circle_conf_threshold: float = CIRCLE_CONF_THRESHOLD,
        apply_undistort: bool = False,
        **circle_kwargs,
    ):
        """
        Args:
            model_path:            YOLO 模型路径 (None=自动查找)
            camera_matrix:         相机内参 (3×3)
            dist_coeffs:           畸变系数
            bucket_height_m:       桶物理高度 (米)
            camera_offset_m:       摄像头距飞机底部高度 (米), Z_C修正用
            yolo_conf:             YOLO 置信度阈值
            yolo_iou:              YOLO NMS IoU 阈值
            imgsz:                 YOLO 输入尺寸
            circle_conf_threshold: 圆检测时使用的最低 YOLO 置信度
            apply_undistort:       是否对每帧做畸变校正
            **circle_kwargs:       传递给 CircleDetector 的参数
        """
        self.circle_conf_threshold = circle_conf_threshold
        self.apply_undistort = apply_undistort

        # 子模块
        self.yolo = YOLODetector(
            model_path=model_path,
            imgsz=imgsz,
            conf=yolo_conf,
            iou=yolo_iou,
        )
        self.circle = CircleDetector(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            bucket_height_m=bucket_height_m,
            camera_offset_m=camera_offset_m,
            **circle_kwargs,
        )

        # 畸变校正映射表
        self._undistort_maps: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # 实时单帧处理
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        alt_rel_m: float = 0.0,
        return_annotated: bool = False,
    ) -> List[dict]:
        """
        对单帧执行: YOLO 推理 → 圆检测 → 直径计算。

        YOLO 模型不可用时，回退到全帧圆检测。

        Args:
            frame:            BGR 图像 (H×W×3)
            alt_rel_m:        当前相对高度 (米), 用于直径计算
            return_annotated: 是否在返回结果中包含标注帧

        Returns:
            每项 dict:
                {
                    "det": dict | None,         # YOLO 检测原始数据
                    "circle": CircleResult,      # 圆检测结果 (None=未检测到)
                    "diameter_m": float,         # 真实直径 (米)
                    "edge_success": bool,
                }
            无检测时返回空列表。

            若 return_annotated=True, 最后一项为 {"annotated_frame": np.ndarray}
        """
        # 畸变校正 (可选)
        if self.apply_undistort:
            if self._undistort_maps is None:
                h, w = frame.shape[:2]
                self._undistort_maps = self.circle.init_undistort_maps(w, h)
            frame = cv2.remap(
                frame, self._undistort_maps[0], self._undistort_maps[1],
                cv2.INTER_LINEAR,
            )

        results = []
        annotated = frame.copy() if return_annotated else None

        # ---- 模式 1: YOLO 模型可用 → YOLO → Circle ----
        if self.yolo.is_available:
            dets = self.yolo.detect(frame)
            bucket_dets = self.yolo.get_class1_detections(
                dets, self.circle_conf_threshold
            )
            print(f"  [YOLO] 总检测={len(dets)}个, "
                  f"桶(cls=1)={len(bucket_dets)}个 "
                  f"(conf≥{self.circle_conf_threshold})")

            for det in bucket_dets:
                bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
                print(f"  [YOLO] 桶: bbox=({det['x1']},{det['y1']},"
                      f"{det['x2']},{det['y2']}) conf={det['conf']:.2f}")
                result = self._detect_and_compute(frame, bbox, alt_rel_m)
                if result is not None:
                    result["det"] = det
                    results.append(result)

                # 可视化叠加
                if annotated is not None:
                    self._annotate(annotated, frame, bbox, result)

        # ---- 模式 2: 无 YOLO 模型 → 全帧圆检测 ----
        else:
            h, w = frame.shape[:2]
            bbox = (0, 0, w, h)
            print(f"  [全帧] 无YOLO模型, 全帧检测 {w}×{h}")
            result = self._detect_and_compute(frame, bbox, alt_rel_m)
            if result is not None:
                result["det"] = None  # 无 YOLO 检测
                results.append(result)

            if annotated is not None:
                self._annotate(annotated, frame, bbox, result)

        if return_annotated:
            results.append({"annotated_frame": annotated})

        return results

    def _detect_and_compute(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
        alt_rel_m: float,
    ) -> Optional[dict]:
        """对单个 bbox 执行圆检测 + 直径计算。"""
        circle_result = self.circle.detect(frame, bbox)

        diameter_m = -1.0
        edge_success = False
        if circle_result is not None:
            diameter_m = self.circle.compute_diameter(
                circle_result.radius_px, alt_rel_m
            )
            edge_success = True
            # 更新 circle_result 中的 diameter_m
            circle_result = CircleResult(
                cx_px=circle_result.cx_px,
                cy_px=circle_result.cy_px,
                radius_px=circle_result.radius_px,
                diameter_px=circle_result.diameter_px,
                diameter_m=diameter_m,
                bbox=circle_result.bbox,
                roi_offset=circle_result.roi_offset,
            )

        return {
            "circle": circle_result,
            "diameter_m": diameter_m,
            "edge_success": edge_success,
        }

    def _annotate(
        self, annotated: np.ndarray, frame: np.ndarray,
        bbox: Tuple[int, int, int, int], result: Optional[dict],
    ):
        """在 annotated 帧上叠加可视化。"""
        roi, rx1, ry1 = self.circle._extract_roi(frame, *bbox)
        edges = self.circle.get_edges(roi)
        circle_r = result["circle"] if result else None
        # overlay 直接修改 annotated (内部 copy)
        # 这里用返回来更新 (overlay 返回新副本)
        new_annotated = self.circle.overlay(
            annotated, rx1, ry1, edges, circle_r
        )
        annotated[:] = new_annotated

    # ------------------------------------------------------------------
    # 预设点计算
    # ------------------------------------------------------------------

    def compute_preset_points(self, alt_rel_m: float):
        """
        计算两个预设投放点在图像上的像素坐标和投放范围半径。

        B1 = 图像中心上方 5.6cm (bottle=1, 对应 15cm 桶)
        B2 = 图像中心下方 5.6cm (bottle=2, 对应 20cm 桶)
        投放范围圆直径 = DROP_ZONE_DIAMETER_M (默认 6cm)

        Args:
            alt_rel_m: 当前相对高度 (米)

        Returns:
            (b1_px, b1_py), (b2_px, b2_py), zone_radius_px
            如果高度无效，返回 (-1,-1), (-1,-1), 0
        """
        water_z = alt_rel_m + CAMERA_OFFSET_M - BUCKET_HEIGHT_M
        if water_z <= 0:
            return (-1, -1), (-1, -1), 0

        fx = self.circle.fx
        fy = self.circle.fy
        cx = self.circle.cx
        cy = self.circle.cy

        offset_px = (WATER_BOTTLE_OFFSET_M * fy) / water_z
        zone_radius_px = int(round((DROP_ZONE_DIAMETER_M / 2.0 * fx) / water_z))

        b1 = (int(round(cx)), int(round(cy - offset_px)))   # 图像上方=北
        b2 = (int(round(cx)), int(round(cy + offset_px)))   # 图像下方=南

        return b1, b2, zone_radius_px
