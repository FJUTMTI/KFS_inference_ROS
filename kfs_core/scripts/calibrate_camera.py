#!/usr/bin/env python3
"""
Camera calibration script using ROS2 camera_calibration package (MonoCalibrator).

Processes a recorded checkerboard video (e.g. assets/calib.mp4) and produces
ost.yaml (compatible with ROS camera_info / camera_calibration output).

Usage examples:
  python3 scripts/calibrate_camera.py \
      --video ../assets/calib.mp4 \
      --size 11x8 \
      --square 0.02 \
      --output ../config/ost.yaml

  # From workspace root (kfs_core/ subdir layout)
  cd realsense_inference
  python3 kfs_core/scripts/calibrate_camera.py --video kfs_core/assets/calib.mp4 --size 11x8 --square 0.02 --output kfs_core/config/ost.yaml

Requirements:
  - ROS2 humble environment (for camera_calibration python package)
  - OpenCV python bindings
"""

import argparse
import sys
import cv2
import numpy as np

try:
    from camera_calibration.calibrator import MonoCalibrator, ChessboardInfo, Patterns
except ImportError as e:
    print("[ERROR] Failed to import from camera_calibration.")
    print("  Make sure you have sourced ROS 2 (or camera_calibration is importable):")
    print("    source /opt/ros/humble/setup.zsh")
    print("  Error:", e)
    sys.exit(1)


def parse_size(s: str):
    """Parse '11x8' or '8x11' -> (n_cols, n_rows) with n_cols >= n_rows for ROS convention."""
    try:
        a, b = s.lower().split("x")
        cols = int(a)
        rows = int(b)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid size format '{s}', expected NxM e.g. 11x8 or 8x11")
    # ROS Calibrator forces n_cols > n_rows internally, but we accept user order
    # and let it normalize. Return larger first as cols to match common usage.
    if cols < rows:
        cols, rows = rows, cols
    return cols, rows


def main():
    parser = argparse.ArgumentParser(description="Calibrate monocular camera from video using ROS2 camera_calibration")
    parser.add_argument("--video", "-v", default="kfs_core/assets/calib.mp4",
                        help="Path to calibration video containing checkerboard (default: kfs_core/assets/calib.mp4)")
    parser.add_argument("--size", "-s", default="11x8",
                        help="Checkerboard interior corners as WxH or HxW (e.g. 11x8). Default matches 11x8 board.")
    parser.add_argument("--square", "-q", type=float, default=0.02,
                        help="Square size in meters (e.g. 0.02 for 20mm). Default 0.02.")
    parser.add_argument("--output", "-o", default="kfs_core/config/ost.yaml",
                        help="Output ost.yaml path (default: kfs_core/config/ost.yaml)")
    parser.add_argument("--name", default="camera",
                        help="Camera name in the yaml (default: camera)")
    parser.add_argument("--max-samples", type=int, default=40,
                        help="Maximum number of samples to collect (default 40, enough for good calib)")
    parser.add_argument("--sample-every", type=int, default=8,
                        help="Sample every N frames (default 8, adjust for your video length)")
    parser.add_argument("--rational-model", action="store_true",
                        help="Use rational polynomial distortion model (more coeffs k1..k6)")
    parser.add_argument("--fix-principal-point", action="store_true",
                        help="Fix principal point at image center during optimization")
    parser.add_argument("--zero-tangent-dist", action="store_true",
                        help="Assume zero tangential distortion (p1=p2=0)")

    args = parser.parse_args()

    n_cols, n_rows = parse_size(args.size)
    square = float(args.square)

    board = ChessboardInfo(n_cols=n_cols, n_rows=n_rows, dim=square)

    # Build flags for calibrateCamera
    flags = 0
    if args.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if args.fix_principal_point:
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if args.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    print("=== Camera Calibration (ROS2 camera_calibration backend) ===")
    print(f"  video       : {args.video}")
    print(f"  board size  : {n_cols}x{n_rows} (interior corners)")
    print(f"  square size : {square} m")
    print(f"  max samples : {args.max_samples}")
    print(f"  sample step : every {args.sample_every} frames")
    print(f"  flags       : {flags} (rational={bool(args.rational_model)}, fix_pp={bool(args.fix_principal_point)}, zero_tan={bool(args.zero_tangent_dist)})")
    print(f"  output      : {args.output}")
    print()

    mc = MonoCalibrator(
        boards=[board],
        flags=flags,
        pattern=Patterns.Chessboard,
        name=args.name,
        checkerboard_flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Video info: {width}x{height} @ {fps:.1f}fps , ~{total_frames} frames\n")

    good_corners = []
    frame_idx = 0
    last_corners = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Decide whether to inspect this frame
        do_inspect = (frame_idx % args.sample_every == 0) or (len(good_corners) < 5)

        if do_inspect:
            ok, corners, ids, b = mc.get_corners(img, refine=True)
            if ok:
                # Simple dedup: skip if very similar to last accepted (pixel motion)
                if last_corners is not None:
                    try:
                        delta = np.linalg.norm(corners - last_corners)
                        if delta < 5.0:  # almost same pose, skip
                            frame_idx += 1
                            continue
                    except Exception:
                        pass

                good_corners.append((corners, ids, b))
                last_corners = corners.copy()
                print(f"  [collect] {len(good_corners):2d}/{args.max_samples}  (frame {frame_idx:5d})")

                if len(good_corners) >= args.max_samples:
                    print("  Reached max samples, stopping collection early.")
                    break

        frame_idx += 1

    cap.release()

    print(f"\nCollected {len(good_corners)} valid corner sets.")

    if len(good_corners) < 10:
        print("[ERROR] Too few good samples (<10). The video may not contain enough varied views of the board,")
        print("        or board size / square size is wrong, or contrast too low.")
        print("        Try:")
        print("          * smaller --sample-every (e.g. 3)")
        print("          * check first frames with board visible using cv2")
        print("          * confirm --size and --square match your physical board")
        sys.exit(2)

    print("Running calibration (this may take a few seconds)...")
    mc.cal_fromcorners(good_corners)
    mc.calibrated = True

    # Print useful report
    try:
        print("\n" + mc.report())
    except Exception as ex:
        print(f"(report unavailable: {ex})")

    # Get the yaml (ost.yaml format)
    yaml_text = mc.yaml()

    # Ensure output dir exists
    import os
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"\n[SUCCESS] Calibration finished.")
    print(f"  Wrote: {os.path.abspath(args.output)}")
    print("\nYou can now use this ost.yaml for camera intrinsics (ROS style).")
    print("Example: load K and D into your USB camera model or publish as camera_info.")

    # Also print key numbers for quick check
    try:
        k = mc.intrinsics
        d = mc.distortion
        print("\nQuick intrinsics:")
        print(f"  fx={k[0,0]:.2f}  fy={k[1,1]:.2f}")
        print(f"  cx={k[0,2]:.2f}  cy={k[1,2]:.2f}")
        print(f"  dist: {np.ravel(d).tolist()}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
