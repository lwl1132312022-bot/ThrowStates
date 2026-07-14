"""
ThrowStates — 视觉识别 + 预设点投放
===================================
流程:
  1. 连接 PX4 → Arm (舵机供电)
  2. 摄像头 → YOLO 桶检测 → Canny 圆检测
  3. 圆心进入预设投放范围 → 释放舵机

预设点:
  B1 (TEST_PRESET=1): 图像中心上方 5.6cm, 对应 15cm 桶, 舵机 AUX7
  B2 (TEST_PRESET=2): 图像中心下方 5.6cm, 对应 20cm 桶, 舵机 AUX8

用法:
  python main.py                    # 默认 TEST_PRESET
  python main.py --preset 1         # 仅测试 B1
  python main.py --preset 2         # 仅测试 B2
  python main.py --preset both      # 两个都测
  python main.py --sim              # 纯视觉测试 (不连飞控，不释放舵机)
  python main.py --height 1.5       # 指定测试高度 (默认 1.5m)
"""

import asyncio
import math
import time
import argparse
from typing import Tuple

import cv2

from config import (
    TEST_PRESET,
    DROP_ALIGN_ALTITUDE_M,
    DROP_SERVO_CHANNEL_1, DROP_SERVO_CHANNEL_2,
    USE_SERVO,
    CAMERA_DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    APPLY_UNDISTORT,
)
from vision import VisionPipeline

# PX4 接口 (仅用于舵机控制)
try:
    from px4_interface import PX4Interface
    HAS_PX4 = True
except ImportError:
    HAS_PX4 = False


# =============================================================================
# 投放区域检查
# =============================================================================

def check_drop_zone(
    circle_cx_px: int, circle_cy_px: int,
    preset_px: int, preset_py: int,
    zone_radius_px: int,
) -> Tuple[bool, float]:
    """圆心到预设点的像素距离是否小于投放范围半径。"""
    dx = circle_cx_px - preset_px
    dy = circle_cy_px - preset_py
    dist_px = math.hypot(dx, dy)
    return dist_px < zone_radius_px, dist_px


# =============================================================================
# 摄像头
# =============================================================================

def open_camera(device: int = CAMERA_DEVICE) -> cv2.VideoCapture:
    """打开 USB 摄像头。"""
    for dev_id in [device, 0, 1, 2]:
        cap = cv2.VideoCapture(dev_id)
        if cap.isOpened():
            break
    else:
        raise RuntimeError("无法打开摄像头，请检查连接。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[摄像头] {w}×{h} @ {fps:.0f}fps")
    return cap


# =============================================================================
# 释放舵机
# =============================================================================

async def release_servo(interface, channel: int, label: str):
    """释放指定舵机通道。"""
    if not USE_SERVO:
        print(f"[投放] {label}: 模拟释放 (USE_SERVO=False)")
        return

    if interface is None:
        print(f"[投放] {label}: 无飞控连接，跳过舵机")
        return

    await interface.set_actuator(channel, 1.0)
    await asyncio.sleep(0.5)
    await interface.set_actuator(channel, -1.0)
    print(f"[投放] {label}: AUX{channel} 已释放")


# =============================================================================
# 主循环 — 视觉检测 + 舵机投放
# =============================================================================

async def run(preset_filter, test_altitude: float, interface=None):
    """
    核心循环: 摄像头 → YOLO → 圆检测 → 预设点判断 → 释放舵机。

    Args:
        preset_filter: "both" | 1 | 2
        test_altitude: 固定测试高度 (米), 用于预设点计算
        interface:     PX4Interface 或 None (纯视觉模式)
    """
    pipeline = VisionPipeline(apply_undistort=APPLY_UNDISTORT)
    cap = open_camera()

    cv2.namedWindow("ThrowStates", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ThrowStates", 1280, 720)

    dropped_b1 = False
    dropped_b2 = False
    frame_count = 0
    fps_start = time.time()

    # 直径匹配容差 (移植自 cuadc_ws/config.py EPSILON_DIAMETER_CM)
    EPSILON_DIAMETER_CM = 2.0
    # B1 对应 15cm 桶, B2 对应 20cm 桶 (移植自 cuadc_ws SearchState)
    B1_EXPECTED_DIAMETER_CM = 15.0
    B2_EXPECTED_DIAMETER_CM = 20.0

    print(f"[任务] 预设点: {preset_filter}, 测试高度: {test_altitude}m")
    print(f"[任务] 畸变校正: {'开' if APPLY_UNDISTORT else '关'}")
    print(f"[任务] 直径匹配: B1={B1_EXPECTED_DIAMETER_CM}±{EPSILON_DIAMETER_CM}cm, "
          f"B2={B2_EXPECTED_DIAMETER_CM}±{EPSILON_DIAMETER_CM}cm")
    print(f"[任务] 舵机通道: B1=AUX{DROP_SERVO_CHANNEL_1}, B2=AUX{DROP_SERVO_CHANNEL_2}")
    print(f"[任务] USE_SERVO={USE_SERVO}, interface={'PX4' if interface else 'None'}")
    print("[任务] 按 'q' 退出\n")

    # 帧率统计
    log_interval = 20  # 每 N 帧打印一次详细日志

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[错误] 帧读取失败")
                break
            frame_count += 1

            # ── 1. 视觉处理 ──
            t0 = time.time()
            results = pipeline.process_frame(
                frame, alt_rel_m=test_altitude, return_annotated=True)
            t_vision = (time.time() - t0) * 1000  # ms

            annotated = frame.copy()
            for r in results:
                if "annotated_frame" in r:
                    annotated = r["annotated_frame"]
                    break

            # ── 2. 计算预设点 ──
            b1, b2, zone_radius_px = pipeline.compute_preset_points(
                test_altitude)
            Z_C = test_altitude + 0.036 - 0.30  # 相机到桶顶距离

            # ── 3. 筛选有效检测 ──
            valid_circles = [
                r for r in results
                if r.get("edge_success") and r["circle"] is not None
            ]
            total_dets = len([r for r in results if "annotated_frame" not in r])
            yolo_dets = len([r for r in results
                            if r.get("det") is not None and "annotated_frame" not in r])

            # 每 N 帧打印一次详细状态
            if frame_count % log_interval == 1:
                elapsed = time.time() - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"\n── 帧 {frame_count} (FPS:{fps:.0f} H:{test_altitude}m "
                      f"Z_C:{Z_C:.2f}m) ──")
                print(f"  视觉耗时: {t_vision:.0f}ms")
                print(f"  YOLO检测: {yolo_dets}个桶, "
                      f"边缘成功: {len(valid_circles)}个圆")
                print(f"  预设点: B1=({b1[0]},{b1[1]}) "
                      f"B2=({b2[0]},{b2[1]}) "
                      f"投放范围半径={zone_radius_px}px")

            # ── 4. 对每个检测到的圆打印详情 ──
            for r in valid_circles:
                circle = r["circle"]
                dia_cm = circle.diameter_m * 100
                if frame_count % log_interval == 1:
                    print(f"  检测圆: 圆心=({circle.cx_px},{circle.cy_px}) "
                          f"半径={circle.radius_px}px "
                          f"直径={dia_cm:.1f}cm "
                          f"(期望B1={B1_EXPECTED_DIAMETER_CM}±{EPSILON_DIAMETER_CM} "
                          f"B2={B2_EXPECTED_DIAMETER_CM}±{EPSILON_DIAMETER_CM})")

            # ── 5. 找最佳匹配圆 ──
            best_b1 = None
            best_b2 = None

            for r in valid_circles:
                circle = r["circle"]
                dia_cm = circle.diameter_m * 100
                cx, cy = circle.cx_px, circle.cy_px

                # B1: 直径匹配 (15cm±2) + 最近距离
                if (preset_filter in ("both", 1)
                        and not dropped_b1 and b1[0] >= 0):
                    dx = cx - b1[0]
                    dy = cy - b1[1]
                    dist = math.hypot(dx, dy)
                    dia_ok = abs(dia_cm - B1_EXPECTED_DIAMETER_CM) <= EPSILON_DIAMETER_CM
                    if frame_count % log_interval == 1:
                        print(f"    B1匹配: dx={dx:+d} dy={dy:+d} "
                              f"距离={dist:.0f}px "
                              f"直径={dia_cm:.1f}cm -> {'OK' if dia_ok else 'NO MATCH'}")
                    if dia_ok and (best_b1 is None or dist < best_b1[1]):
                        best_b1 = (circle, dist, dia_cm)

                # B2: 直径匹配 (20cm±2) + 最近距离
                if (preset_filter in ("both", 2)
                        and not dropped_b2 and b2[0] >= 0):
                    dx = cx - b2[0]
                    dy = cy - b2[1]
                    dist = math.hypot(dx, dy)
                    dia_ok = abs(dia_cm - B2_EXPECTED_DIAMETER_CM) <= EPSILON_DIAMETER_CM
                    if frame_count % log_interval == 1:
                        print(f"    B2匹配: dx={dx:+d} dy={dy:+d} "
                              f"距离={dist:.0f}px "
                              f"直径={dia_cm:.1f}cm -> {'OK' if dia_ok else 'NO MATCH'}")
                    if dia_ok and (best_b2 is None or dist < best_b2[1]):
                        best_b2 = (circle, dist, dia_cm)

            # ── 6. 投放判断 ──
            in_zone_b1 = False
            in_zone_b2 = False

            # B1
            if best_b1 is not None:
                circle_b1, dist_b1, dia_b1 = best_b1
                if dist_b1 < zone_radius_px:
                    in_zone_b1 = True
                    if not dropped_b1:
                        print(f"\n{'='*50}")
                        print(f"[投放] B1 触发!")
                        print(f"  圆心: ({circle_b1.cx_px},{circle_b1.cy_px})")
                        print(f"  预设: B1({b1[0]},{b1[1]})")
                        print(f"  距离: {dist_b1:.0f}px < {zone_radius_px}px OK")
                        print(f"  直径: {dia_b1:.1f}cm "
                              f"(期望 {B1_EXPECTED_DIAMETER_CM}+/-{EPSILON_DIAMETER_CM}cm) OK")
                        print(f"  高度: {test_altitude}m, Z_C={Z_C:.2f}m")
                        print(f"  舵机: AUX{DROP_SERVO_CHANNEL_1}")
                        print(f"{'='*50}\n")
                        dropped_b1 = True
                        await release_servo(
                            interface, DROP_SERVO_CHANNEL_1, "B1(AUX7)")

            # B2
            if best_b2 is not None:
                circle_b2, dist_b2, dia_b2 = best_b2
                if dist_b2 < zone_radius_px:
                    in_zone_b2 = True
                    if not dropped_b2:
                        print(f"\n{'='*50}")
                        print(f"[投放] B2 触发!")
                        print(f"  圆心: ({circle_b2.cx_px},{circle_b2.cy_px})")
                        print(f"  预设: B2({b2[0]},{b2[1]})")
                        print(f"  距离: {dist_b2:.0f}px < {zone_radius_px}px OK")
                        print(f"  直径: {dia_b2:.1f}cm "
                              f"(期望 {B2_EXPECTED_DIAMETER_CM}+/-{EPSILON_DIAMETER_CM}cm) OK")
                        print(f"  高度: {test_altitude}m, Z_C={Z_C:.2f}m")
                        print(f"  舵机: AUX{DROP_SERVO_CHANNEL_2}")
                        print(f"{'='*50}\n")
                        dropped_b2 = True
                        await release_servo(
                            interface, DROP_SERVO_CHANNEL_2, "B2(AUX8)")

            # ── 6. 在图上标记: 圆心进入范围 → 绿色高亮 ──
            hit_circle = best_b1[0] if in_zone_b1 else (best_b2[0] if in_zone_b2 else None)
            if hit_circle is not None:
                GREEN = (0, 255, 0)
                cx, cy = hit_circle.cx_px, hit_circle.cy_px
                r_px = hit_circle.radius_px
                cv2.circle(annotated, (cx, cy), r_px, GREEN, 3)
                cv2.circle(annotated, (cx, cy), 6, GREEN, -1)
                cross = max(r_px // 2, 5)
                cv2.line(annotated, (cx - cross, cy),
                         (cx + cross, cy), GREEN, 2)
                cv2.line(annotated, (cx, cy - cross),
                         (cx, cy + cross), GREEN, 2)
                if in_zone_b1:
                    cv2.line(annotated, (cx, cy), b1, GREEN, 2)
                if in_zone_b2:
                    cv2.line(annotated, (cx, cy), b2, (0, 220, 0), 2)

            # 同时对每个检测到的圆标注直径 (调试用)
            for r in valid_circles:
                c = r["circle"]
                if c is not hit_circle and c.diameter_m > 0:
                    dia_label = f"{c.diameter_m * 100:.1f}cm"
                    cv2.putText(annotated, dia_label,
                                (c.cx_px + c.radius_px + 3,
                                 c.cy_px + c.radius_px + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (200, 200, 200), 1)

            # ── 4. 叠加预设点可视化 ──
            if b1[0] >= 0:
                # B1
                if preset_filter in ("both", 1):
                    if dropped_b1:
                        color = (100, 100, 100)
                    elif in_zone_b1:
                        color = (0, 255, 0)      # 绿色 = 当前在范围内
                    else:
                        color = (255, 255, 0)     # 青色 = 等待中
                    cv2.circle(annotated, b1, zone_radius_px, color, 2)
                    cv2.circle(annotated, b1, 5, color, -1)
                    label = ("B1 DONE" if dropped_b1
                             else "B1 HIT!" if in_zone_b1 else "B1")
                    cv2.putText(annotated, label, (b1[0] + 10, b1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # B2
                if preset_filter in ("both", 2):
                    if dropped_b2:
                        color = (100, 100, 100)
                    elif in_zone_b2:
                        color = (0, 255, 0)
                    else:
                        color = (0, 165, 255)
                    cv2.circle(annotated, b2, zone_radius_px, color, 2)
                    cv2.circle(annotated, b2, 5, color, -1)
                    label = ("B2 DONE" if dropped_b2
                             else "B2 HIT!" if in_zone_b2 else "B2")
                    cv2.putText(annotated, label, (b2[0] + 10, b2[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # ── 5. 画面中央大字提示 ──
            if in_zone_b1:
                cv2.putText(annotated, "B1 IN ZONE!",
                            (annotated.shape[1] // 2 - 120, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if in_zone_b2:
                cv2.putText(annotated, "B2 IN ZONE!",
                            (annotated.shape[1] // 2 - 120, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # ── 6. 状态栏 ──
            if frame_count % 5 == 0:
                elapsed = time.time() - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0

            parts = [f"FPS:{fps:.0f}", f"H:{test_altitude}m"]
            if preset_filter in ("both", 1):
                parts.append("B1:HIT" if in_zone_b1
                             else "B1:DONE" if dropped_b1
                             else "B1:wait")
            if preset_filter in ("both", 2):
                parts.append("B2:HIT" if in_zone_b2
                             else "B2:DONE" if dropped_b2
                             else "B2:wait")
            cv2.putText(annotated, " | ".join(parts),
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
            cv2.putText(annotated, "Q=quit",
                        (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (100, 100, 100), 1)

            # ── 7. 显示 ──
            cv2.imshow("ThrowStates", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[信息] 用户退出")
                break

    except KeyboardInterrupt:
        print("\n[信息] 用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[统计] {frame_count} 帧, "
              f"B1={'OK' if dropped_b1 else 'NO'}, "
              f"B2={'OK' if dropped_b2 else 'NO'}")


# =============================================================================
# 入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ThrowStates — 视觉识别 + 预设点投放")
    parser.add_argument(
        "--preset", type=str, default=str(TEST_PRESET),
        choices=["1", "2", "both"],
        help=f"预设点: 1=B1, 2=B2, both=两个 (默认: {TEST_PRESET})")
    parser.add_argument(
        "--sim", action="store_true",
        help="纯视觉模式 (不连飞控，不释放舵机)")
    parser.add_argument(
        "--height", type=float, default=DROP_ALIGN_ALTITUDE_M,
        help=f"测试高度/米 (默认: {DROP_ALIGN_ALTITUDE_M})")
    parser.add_argument(
        "--test-servo", type=int, default=0, metavar="CH",
        help="舵机测试: 只连飞控, 循环测试指定通道 (无视视觉)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preset_filter = "both" if args.preset == "both" else int(args.preset)

    # 舵机测试模式 — 绕过视觉，直接测舵机
    if args.test_servo:
        async def test_servo():
            if not HAS_PX4:
                print("[错误] mavsdk 未安装")
                return
            interface = PX4Interface()
            await interface.connect_and_setup()
            ch = args.test_servo
            print(f"\n[舵机测试] 通道 AUX{ch}, 循环 3 次 (开1s/关1s)")
            for i in range(3):
                print(f"  第 {i+1}/3 次: 打开...")
                await interface.set_actuator(ch, 1.0)
                await asyncio.sleep(1.0)
                print(f"  第 {i+1}/3 次: 关闭...")
                await interface.set_actuator(ch, -1.0)
                await asyncio.sleep(1.0)
            print("[舵机测试] 完成")

        asyncio.run(test_servo())
        sys.exit(0)

    if args.sim:
        # 纯视觉测试，不连飞控
        asyncio.run(run(preset_filter, args.height, interface=None))
    else:
        # 连接飞控，仅用于舵机控制 (不上锁，AUX 独立供电)
        async def main_with_px4():
            if not HAS_PX4:
                print("[错误] mavsdk 未安装，请用 --sim 模式")
                return

            interface = PX4Interface()
            try:
                await interface.connect_and_setup()
            except Exception as e:
                print(f"[错误] PX4 连接失败: {e}")
                print("[信息] 回退到纯视觉模式")
                await run(preset_filter, args.height, interface=None)
                return

            await run(preset_filter, args.height, interface=interface)

        asyncio.run(main_with_px4())
