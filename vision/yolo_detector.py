#!/usr/bin/env python3
"""
YOLO 检测器模块 — 桶检测 (TensorRT / PyTorch)
===============================================
封装 Ultralytics YOLO 模型加载与推理。

用法:
    from vision.yolo_detector import YOLODetector

    detector = YOLODetector("models/yolov11n_800_best_FP16.engine")
    dets = detector.detect(frame)  # → [{x1,y1,x2,y2,conf,cls,name}, ...]

如果没有模型文件，YOLODetector 会以无模型模式运行，
此时 detect() 返回空列表，由 pipeline 回退到全帧圆检测。

参考: cuadc_ws/src/vision/yolo_detector.py
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


# 默认模型路径 — 相对于本文件所在目录的 models/
_DEFAULT_MODEL_NAME = "yolov11n_800_best_FP16.engine"
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"


class YOLODetector:
    """YOLO 目标检测器 (TensorRT / PyTorch)"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        imgsz: int = 800,
        conf: float = 0.5,
        iou: float = 0.45,
    ):
        """
        Args:
            model_path: 模型文件路径 (.engine / .pt / .onnx).
                        默认: vision/models/yolov11n_800_best_FP16.engine
                        设为 None 且找不到默认模型时，进入无模型模式。
            imgsz:     模型输入尺寸 (默认 800)
            conf:      置信度阈值 (0~1)
            iou:       NMS IoU 阈值
        """
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.model = None
        self._names: Dict[int, str] = {}
        self._model_available = False

        # 解析模型路径
        if model_path is None:
            model_path = str(_DEFAULT_MODEL_DIR / _DEFAULT_MODEL_NAME)
        self.model_path = self._resolve_model_path(model_path)
        self._load()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _resolve_model_path(self, path: str) -> str:
        """尝试自动补全模型文件扩展名 (.engine → .pt → .onnx)"""
        if os.path.exists(path):
            return path

        base = path
        for ext in [".engine", ".pt", ".onnx"]:
            for old_ext in [".engine", ".pt", ".onnx"]:
                if base.endswith(old_ext):
                    base = base[: -len(old_ext)]
                    break
            candidate = base + ext
            if os.path.exists(candidate):
                return candidate

        return path

    def _load(self):
        """加载模型。如果模型文件不存在，进入无模型模式。"""
        mp = self.model_path
        if not os.path.exists(mp):
            print(f"[YOLO] 模型文件未找到: {mp}")
            print(f"[YOLO] 进入无模型模式 — 将使用全帧圆检测作为回退")
            print(f"[YOLO] 如需 YOLO 检测，请将模型放到 vision/models/ 目录")
            self._model_available = False
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(mp)
            backend = "TensorRT" if mp.endswith(".engine") else "PyTorch"
            if hasattr(self.model, "names"):
                self._names = self.model.names
            self._model_available = True
            print(f"[YOLO] 模型已加载: {mp} [{backend}]")
        except ImportError:
            print(f"[YOLO] ultralytics 未安装，进入无模型模式")
            self._model_available = False
        except Exception as e:
            print(f"[YOLO] 模型加载失败 ({e})，进入无模型模式")
            self._model_available = False

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        对单帧图像进行目标检测。

        Args:
            frame: BGR 图像 (numpy ndarray, H×W×3)

        Returns:
            检测结果列表, 每项为:
                {
                    "x1": int, "y1": int, "x2": int, "y2": int,  # 边界框
                    "conf": float,                                   # 置信度
                    "cls": int,                                      # 类别 ID
                    "name": str,                                     # 类别名称
                }
            无检测或无模型时返回空列表。
        """
        if not self._model_available or self.model is None:
            return []

        results = self.model(
            frame,
            verbose=False,
            device=0,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            half=True,  # FP16
        )

        dets = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    dets.append(
                        {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "conf": round(float(box.conf[0]), 3),
                            "cls": cls_id,
                            "name": self._names.get(cls_id, "?"),
                        }
                    )
        return dets

    # ------------------------------------------------------------------
    # 属性与方法
    # ------------------------------------------------------------------

    @property
    def names(self) -> Dict[int, str]:
        """类别名称映射 {cls_id: name}"""
        return self._names

    @property
    def is_available(self) -> bool:
        """模型是否可用"""
        return self._model_available

    def get_class1_detections(
        self, dets: List[dict], conf_threshold: Optional[float] = None
    ) -> List[dict]:
        """
        过滤出 class=1 (bucket) 且置信度达标的检测, 按置信度降序排列。

        Args:
            dets:          detect() 返回的检测列表
            conf_threshold: 置信度阈值, 默认使用实例的 self.conf

        Returns:
            过滤并排序后的 bucket 检测列表
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf
        bucket_dets = [
            d for d in dets if d["cls"] == 1 and d["conf"] >= threshold
        ]
        bucket_dets.sort(key=lambda d: d["conf"], reverse=True)
        return bucket_dets
