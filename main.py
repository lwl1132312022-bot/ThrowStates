"""
ThrowStates — 视觉识别 + 预设点投放 主程序
===========================================
流程:
  1. 连接 PX4 → Arm → 起飞 → Offboard 模式
  2. 视觉循环: 摄像头捕获 → YOLO 桶检测 → 圆检测
  3. 检查圆心是否进入预设投放范围
  4. 进入范围 → 释放舵机 → 降落

预设点:
  B1 (TEST_PRESET=1): 图像中心上方 5.6cm, 对应 15cm 桶, 舵机 AUX7
  B2 (TEST_PRESET=2): 图像中心下方 5.6cm, 对应 20cm 桶, 舵机 AUX8

用法:
  python main.py                    # 默认使用 TEST_PRESET 配置
  python main.py --preset 1         # 仅测试 B1
  python main.py --preset 2         # 仅测试 B2
  python main.py --preset both      # 两个都测
  python main.py --sim              # 纯视觉测试模式 (不连接飞控)
"""

import asyncio
import math
import sys
import time
import argparse
from typing import Optional, Tuple

import cv2

from config import (
    TEST_PRESET,
    CRUISE_ALTITUDE_M, DROP_ALIGN_ALTITUDE_M,
    FSM_LOOP_HZ, ARRIVAL_THRESHOLD_M,
    DROP_SERVO_CHANNEL_1, DROP_SERVO_CHANNEL_2,
    USE_SERVO,
    CAMERA_DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
)
from vision import VisionPipeline, CircleDetector
from pinhole_model import DEFAULT_CAMERA

# PX4 接口 — 可选导入 (纯视觉测试模式不需要)
try:
    from px4_interface import PX4Interface
    HAS_PX4 = True
except ImportError as e:
    print(f"[警告] PX4 接口不可用: {e}")
    HAS_PX4 = False


# =============================================================================
# 投放区域检查
# =============================================================================

def check_drop_zone(
    circle_cx_px: int,
    circle_cy_px: int,
    preset_px: int,
    preset_py: int,
    zone_radius_px: int,
) -> Tuple[bool, float]:
    """
    检查检测到的圆心是否在预设投放圆的范围内。

    Args:
        circle_cx_px, circle_cy_px: 检测到的圆心像素坐标
        preset_px, preset_py:       预设投放点像素坐标
        zone_radius_px:             投放范围圆半径 (像素)

    Returns:
        (is_in_zone: bool, distance_px: float)
    """
    dx = circle_cx_px - preset_px
    dy = circle_cy_px - preset_py
    dist_px = math.hypot(dx, dy)
    return dist_px < zone_radius_px, dist_px


# =============================================================================
# 摄像头初始化
# =============================================================================

def open_camera(device: int = CAMERA_DEVICE) -> cv2.VideoCapture:
    """打开摄像头。"""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        # 尝试不同的设备索引
        for dev_id in [0, 1, 2]:
            cap = cv2.VideoCapture(dev_id)
            if cap.isOpened():
                break
        else:
            raise RuntimeError(
                f"无法打开摄像头 (尝试了设备 0, 1, 2)。"
                f"请检查摄像头连接或使用 --sim 模式。")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[摄像头] 已打开: {actual_w}×{actual_h} @ {actual_fps:.0f}fps")

    return cap


# =============================================================================
# 纯视觉测试模式 (不连接飞控)
# =============================================================================

async def run_vision_test(preset_filter):
    """
    纯视觉测试模式 — 仅测试摄像头 + 视觉识别 + 预设点检测。
    不连接飞控，检测到目标后在画面上标注但不释放舵机。
    """
    print("\n" + "=" * 60)
    print("  纯视觉测试模式")
    print(f"  预设点: {preset_filter}")
    print("  按 'q' 退出, 按 's' 截图")
    print("=" * 60 + "\n")

    # 初始化
    pipeline = VisionPipeline()

    try:
        cap = open_camera()
    except RuntimeError as e:
        print(f"[错误] {e}")
        print("[信息] 请连接摄像头后重试")
        return

    # 显示窗口
    cv2.namedWindow("ThrowStates Vision Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ThrowStates Vision Test", 1280, 720)

    # 使用固定测试高度
    test_altitude = DROP_ALIGN_ALTITUDE_M

    frame_count = 0
    detect_count = 0
    fps_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[错误] 帧读取失败")
                break

            frame_count += 1

            # 视觉处理
            results = pipeline.process_frame(
                frame, alt_rel_m=test_altitude, return_annotated=True)

            # 提取标注帧
            annotated = frame.copy()
            for r in results:
                if "annotated_frame" in r:
                    annotated = r["annotated_frame"]
                    break

            # 计算预设点
            b1, b2, zone_radius_px = pipeline.compute_preset_points(
                test_altitude)

            # 检查每个检测结果
            for r in results:
                if not r.get("edge_success"):
                    continue
                circle = r["circle"]
                if circle is None:
                    continue

                # 检查 B1
                if preset_filter in ("both", 1) and b1[0] >= 0:
                    in_zone, dist = check_drop_zone(
                        circle.cx_px, circle.cy_px,
                        b1[0], b1[1], zone_radius_px)
                    if in_zone:
                        detect_count += 1
                        cv2.putText(
                            annotated,
                            f"B1 IN ZONE! dist={dist:.0f}px",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                # 检查 B2
                if preset_filter in ("both", 2) and b2[0] >= 0:
                    in_zone, dist = check_drop_zone(
                        circle.cx_px, circle.cy_px,
                        b2[0], b2[1], zone_radius_px)
                    if in_zone:
                        detect_count += 1
                        cv2.putText(
                            annotated,
                            f"B2 IN ZONE! dist={dist:.0f}px",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

            # 绘制预设点和投放范围圆
            if b1[0] >= 0:
                if preset_filter in ("both", 1):
                    cv2.circle(annotated, b1, zone_radius_px,
                               (255, 255, 0), 1)  # 青色圆
                    cv2.circle(annotated, b1, 5, (255, 255, 0), -1)
                    cv2.putText(annotated, "B1", (b1[0] + 10, b1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)

                if preset_filter in ("both", 2):
                    cv2.circle(annotated, b2, zone_radius_px,
                               (255, 165, 0), 1)  # 橙色圆
                    cv2.circle(annotated, b2, 5, (255, 165, 0), -1)
                    cv2.putText(annotated, "B2", (b2[0] + 10, b2[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 165, 0), 1)

            # 帧信息
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(annotated,
                            f"FPS: {fps:.1f} | Alt: {test_altitude}m | "
                            f"Detections: {detect_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1)

            # 显示
            cv2.imshow("ThrowStates Vision Test", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"[截图] 已保存: {filename}")

    except KeyboardInterrupt:
        print("\n[信息] 用户中断")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[统计] 处理 {frame_count} 帧, "
              f"检测到 {detect_count} 次进入投放范围")


# =============================================================================
# 完整任务模式 (连接飞控)
# =============================================================================

async def run_mission(preset_filter):
    """
    完整任务模式 — 连接 PX4 飞控，执行投放任务。

    流程:
      connect → arm → takeoff → offboard → vision loop → servo → land
    """
    if not HAS_PX4:
        print("[错误] PX4 接口不可用 (mavsdk 未安装?)")
        print("[信息] 请使用 --sim 模式进行纯视觉测试")
        return

    print("\n" + "=" * 60)
    print("  ThrowStates 投放任务")
    print(f"  预设点: {preset_filter}")
    print("=" * 60 + "\n")

    # ── 1. 初始化 ─────────────────────────────────────
    interface = PX4Interface()
    pipeline = VisionPipeline()

    # ── 2. 连接 + Arm + 起飞 ──────────────────────────
    try:
        await interface.connect_and_setup()
        await interface.arm()
        await interface.takeoff(CRUISE_ALTITUDE_M)
    except Exception as e:
        print(f"[致命] 起飞前阶段失败: {e}")
        return

    # ── 3. 进入 Offboard 模式 ──────────────────────────
    try:
        await interface.switch_to_offboard()
    except Exception as e:
        print(f"[致命] Offboard 切换失败: {e}")
        await interface.land()
        return

    # 记录悬停位置
    hover_pos = await interface.get_position_ned()
    print(f"[信息] 悬停位置: "
          f"N({hover_pos.north_m:.1f}) E({hover_pos.east_m:.1f}) "
          f"高度 {CRUISE_ALTITUDE_M}m")

    # ── 4. 打开摄像头 ─────────────────────────────────
    try:
        cap = open_camera()
    except RuntimeError as e:
        print(f"[错误] {e}")
        await interface.land()
        await interface.disarm()
        return

    # ── 5. 视觉识别循环 ───────────────────────────────
    print("\n[任务] 开始视觉识别循环...")
    print(f"[任务] 等待桶口进入预设投放范围...\n")

    loop_interval = 1.0 / FSM_LOOP_HZ
    no_vision_count = 0
    drop_attempts = 0
    dropped_b1 = False
    dropped_b2 = False
    mission_done = False

    # 显示窗口
    try:
        cv2.namedWindow("ThrowStates Mission", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ThrowStates Mission", 1280, 720)
    except Exception:
        pass  # 无头模式

    try:
        while not mission_done:
            loop_start = time.monotonic()

            # ── 健康检查 ──
            if not await interface.global_guard_check():
                print("[紧急] 健康检查失败，触发降落")
                break

            # ── 捕获帧 ──
            ret, frame = cap.read()
            if not ret:
                no_vision_count += 1
                if no_vision_count > 50:
                    print("[警告] 连续 50 帧无数据")
                await asyncio.sleep(loop_interval)
                continue
            no_vision_count = 0

            # ── 获取当前高度 ──
            alt = await interface.get_altitude()

            # ── 视觉处理 ──
            results = pipeline.process_frame(
                frame, alt_rel_m=alt, return_annotated=True)

            annotated = frame.copy()
            for r in results:
                if "annotated_frame" in r:
                    annotated = r["annotated_frame"]
                    break

            # ── 计算预设点 ──
            b1, b2, zone_radius_px = pipeline.compute_preset_points(alt)

            # ── 检查投放区域 ──
            for r in results:
                if not r.get("edge_success"):
                    continue
                circle = r["circle"]
                if circle is None:
                    continue

                drop_attempts += 1

                # --- 检查 B1 ---
                if (preset_filter in ("both", 1)
                        and not dropped_b1 and b1[0] >= 0):
                    in_zone, dist_px = check_drop_zone(
                        circle.cx_px, circle.cy_px,
                        b1[0], b1[1], zone_radius_px)
                    if in_zone:
                        print(f"\n[投放] B1 进入投放范围! "
                              f"距离={dist_px:.0f}px < {zone_radius_px}px")
                        dropped_b1 = True
                        if USE_SERVO:
                            await interface.set_actuator(
                                DROP_SERVO_CHANNEL_1, 1.0)
                            await asyncio.sleep(0.5)
                            await interface.set_actuator(
                                DROP_SERVO_CHANNEL_1, -1.0)
                            print(f"[投放] B1: AUX{DROP_SERVO_CHANNEL_1} 已释放")
                        else:
                            print(f"[投放] B1: 模拟释放 "
                                  f"(USE_SERVO=False)")
                        if preset_filter != "both":
                            mission_done = True

                # --- 检查 B2 ---
                if (preset_filter in ("both", 2)
                        and not dropped_b2 and b2[0] >= 0):
                    in_zone, dist_px = check_drop_zone(
                        circle.cx_px, circle.cy_px,
                        b2[0], b2[1], zone_radius_px)
                    if in_zone:
                        print(f"\n[投放] B2 进入投放范围! "
                              f"距离={dist_px:.0f}px < {zone_radius_px}px")
                        dropped_b2 = True
                        if USE_SERVO:
                            await interface.set_actuator(
                                DROP_SERVO_CHANNEL_2, 1.0)
                            await asyncio.sleep(0.5)
                            await interface.set_actuator(
                                DROP_SERVO_CHANNEL_2, -1.0)
                            print(f"[投放] B2: AUX{DROP_SERVO_CHANNEL_2} 已释放")
                        else:
                            print(f"[投放] B2: 模拟释放 "
                                  f"(USE_SERVO=False)")
                        if preset_filter != "both":
                            mission_done = True

                # both 模式下两个都投放完成
                if preset_filter == "both" and dropped_b1 and dropped_b2:
                    mission_done = True

            # ── 绘制预设点 ──
            if b1[0] >= 0:
                if preset_filter in ("both", 1) and not dropped_b1:
                    color = (0, 255, 255)  # 黄色 = 等待中
                else:
                    color = (100, 100, 100)  # 灰色 = 已完成
                cv2.circle(annotated, b1, zone_radius_px, color, 1)
                cv2.circle(annotated, b1, 5, color, -1)
                cv2.putText(annotated, f"B1 {'DONE' if dropped_b1 else ''}",
                            (b1[0] + 10, b1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if preset_filter in ("both", 2) and not dropped_b2:
                    color = (0, 165, 255)  # 橙色 = 等待中
                else:
                    color = (100, 100, 100)  # 灰色 = 已完成
                cv2.circle(annotated, b2, zone_radius_px, color, 1)
                cv2.circle(annotated, b2, 5, color, -1)
                cv2.putText(annotated, f"B2 {'DONE' if dropped_b2 else ''}",
                            (b2[0] + 10, b2[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # ── 状态信息 ──
            status_parts = [f"Alt: {alt:.1f}m"]
            if preset_filter in ("both", 1):
                status_parts.append(
                    f"B1: {'DONE' if dropped_b1 else 'waiting'}")
            if preset_filter in ("both", 2):
                status_parts.append(
                    f"B2: {'DONE' if dropped_b2 else 'waiting'}")
            status_parts.append(f"Attempts: {drop_attempts}")
            cv2.putText(annotated, " | ".join(status_parts),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

            cv2.putText(annotated,
                        "Press 'q' to abort",
                        (10, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 100, 100), 1)

            # ── 显示 ──
            try:
                cv2.imshow("ThrowStates Mission", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[信息] 用户中断")
                    break
            except Exception:
                pass

            # ── 维持悬停 ──
            interface.update_setpoint(hover_pos)

            # ── 周期性状态打印 ──
            if drop_attempts > 0 and drop_attempts % 50 == 0:
                print(f"[任务] 等待桶口进入投放范围... "
                      f"(第{drop_attempts}次检测, 高度={alt:.1f}m)")

            # ── 帧率控制 ──
            elapsed = time.monotonic() - loop_start
            sleep_time = loop_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[信息] 用户中断")
    except Exception as e:
        print(f"\n[错误] 视觉循环异常: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ── 6. 降落 ───────────────────────────────────────
    print("\n[任务] 任务完成，开始降落...")

    if dropped_b1 or dropped_b2:
        result_str = []
        if dropped_b1:
            result_str.append("B1")
        if dropped_b2:
            result_str.append("B2")
        print(f"[任务] 成功投放: {', '.join(result_str)}")
    else:
        print("[任务] 未触发投放")

    try:
        await interface.land()
        # 等待着陆
        await asyncio.sleep(5)
        await interface.disarm()
    except Exception as e:
        print(f"[警告] 降落/关停异常: {e}")
        try:
            await interface.disarm()
        except Exception:
            pass

    print("[任务] 完成")


# =============================================================================
# 入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ThrowStates — 视觉识别 + 预设点投放")
    parser.add_argument(
        "--preset", type=str, default=str(TEST_PRESET),
        choices=["1", "2", "both"],
        help=f"测试预设点: 1=B1(上方), 2=B2(下方), both=两个 "
             f"(默认: {TEST_PRESET})")
    parser.add_argument(
        "--sim", action="store_true",
        help="纯视觉测试模式 (不连接飞控，仅测试识别)")
    parser.add_argument(
        "--model", type=str, default=None,
        help="YOLO 模型路径 (默认自动查找)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 解析 TEST_PRESET
    if args.preset == "both":
        preset_filter = "both"
    else:
        preset_filter = int(args.preset)

    if args.sim:
        asyncio.run(run_vision_test(preset_filter))
    else:
        asyncio.run(run_mission(preset_filter))
