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
      --output ../config/ost.yaml \
      --scale 1.0

  # From workspace root (kfs_core/ subdir layout)
  cd realsense_inference
  python3 kfs_core/scripts/calibrate_camera.py \
      --video kfs_core/assets/calib.mp4 \
      --size 11x8 \
      --square 0.02 \
      --output kfs_core/config/ost.yaml \
      --scale 1.0 \
      --sample-every 15

# Use assets example videos for verification of distortion correction (using existing ost.yaml):
python3 kfs_core/scripts/calibrate_camera.py \
  --output kfs_core/config/ost.yaml \
  --undistort-video kfs_core/assets/001.mp4
# (will auto-detect existing ost and produce side-by-side RAW|UNDISTORTED mp4)

# Or explicitly:
python3 -c '
import sys
sys.path.insert(0, "kfs_core/scripts")
import calibrate_camera as cc
cc.demo_undistort("kfs_core/assets/002.mp4", "kfs_core/config/ost.yaml",
                  "/tmp/002_side_by_side.mp4", num_frames=120, side_by_side=True)
'

Requirements:
  - ROS2 humble environment (for camera_calibration python package)
  - OpenCV python bindings
"""

import argparse
import sys
import os
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
    parser.add_argument("--max-samples", type=int, default=20,
                        help="Maximum number of samples to collect (default 20, enough for good calib; more is slower)")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Sample every N frames (default 10; use 5-8 for dense on long videos, higher=15+ for speed)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Processing scale factor (default 1.0 = native resolution of video, e.g. 1920x1080). "
                             "Use 0.5 for ~4x faster on high-res videos (produces scaled intrinsics).")
    parser.add_argument("--rational-model", action="store_true",
                        help="Use rational polynomial distortion model (more coeffs k1..k6)")
    parser.add_argument("--fix-principal-point", action="store_true",
                        help="Fix principal point at image center during optimization")
    parser.add_argument("--zero-tangent-dist", action="store_true",
                        help="Assume zero tangential distortion (p1=p2=0)")
    parser.add_argument("--demo-undistort", action="store_true",
                        help="After writing ost.yaml, also run a quick undistort demo on the input video "
                             "(writes _undistorted.mp4 next to output). Useful to visually verify correction.")
    parser.add_argument("--undistort-video",
                        help="Separate input video for undistortion preview using assets example videos "
                             "(e.g. kfs_core/assets/001.mp4 or 002.mp4). Produces side-by-side RAW|UNDIST video for easy visual verification.")
    parser.add_argument("--undistort-output",
                        help="Explicit output path for the (side-by-side) undistorted preview video.")

    args = parser.parse_args()

    # Pure "apply undistort to a video using existing ost" mode (great for verifying on assets/ *.mp4)
    if args.undistort_video and os.path.exists(args.output):
        in_vid = args.undistort_video
        default_out = os.path.join("/tmp", 
                         os.path.splitext(os.path.basename(in_vid))[0] + "_side_by_side.mp4")
        out_vid = args.undistort_output or default_out
        try:
            demo_undistort(in_vid, args.output, out_vid, num_frames=150, side_by_side=True)
            print("[INFO] Pure undistort verification done (no re-calibration because ost already exists).")
            return
        except Exception as ex:
            print(f"[DEMO] failed: {ex}")
            # fall through to normal if wanted

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
    print(f"  scale       : {args.scale}x (native if 1.0)")
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

    calib_scale = float(args.scale)
    if calib_scale != 1.0:
        print(f"[INFO] Processing at {calib_scale}x scale (resulting ost.yaml will be for ~{int(width*calib_scale)}x{int(height*calib_scale)} resolution). "
              f"Use --scale 1.0 for native {width}x{height}.\n")

    good_corners = []
    frame_idx = 0
    last_corners = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Downscale for calibration speed (affects output resolution of ost.yaml)
        if calib_scale != 1.0:
            img = cv2.resize(img, (0, 0), fx=calib_scale, fy=calib_scale)

        # Decide whether to inspect this frame (use downsample for speed)
        do_inspect = (frame_idx % args.sample_every == 0) or (len(good_corners) < 5)

        if do_inspect:
            # Use downsample_and_detect for fast detection on VGA + full-res refine
            scrib, corners, dsc, ids, b, scales = mc.downsample_and_detect(img)
            if corners is not None:
                # Simple dedup: skip if very similar to last accepted (pixel motion)
                if last_corners is not None:
                    try:
                        delta = np.linalg.norm(corners - last_corners)
                        thresh = 15.0 if calib_scale >= 0.9 else 5.0
                        if delta < thresh:  # almost same pose, skip (scale-aware)
                            frame_idx += 1
                            continue
                    except Exception:
                        pass

                good_corners.append((corners, ids, b))
                last_corners = corners.copy()
                print(f"  [collect] {len(good_corners):2d}/{args.max_samples}  (frame {frame_idx:5d})")

                # Set size on first good sample (required by cal_fromcorners; (width, height))
                if len(good_corners) == 1:
                    mc.size = (img.shape[1], img.shape[0])

                if len(good_corners) >= args.max_samples:
                    print("  Reached max samples, stopping collection early.")
                    break

        frame_idx += 1

    cap.release()

    print(f"\nCollected {len(good_corners)} valid corner sets.")

    if len(good_corners) < 8:
        print("[ERROR] Too few good samples (<8). The video may not contain enough varied views of the board,")
        print("        or board size / square size is wrong, or contrast too low.")
        print("        Try:")
        print("          * smaller --sample-every (e.g. 8) to inspect more frames")
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

    if args.demo_undistort or args.undistort_video or args.undistort_output:
        in_vid = args.undistort_video or args.video
        if args.undistort_output:
            demo_out = args.undistort_output
        else:
            out_dir = os.path.dirname(args.output) or "."
            base = os.path.splitext(os.path.basename(in_vid))[0]
            suffix = "_sidebyside.mp4" if (args.undistort_video or args.demo_undistort) else "_undistorted.mp4"
            demo_out = os.path.join(out_dir, base + suffix)
        use_sbs = bool(args.undistort_video or args.demo_undistort)
        try:
            demo_undistort(in_vid, args.output, demo_out, num_frames=120, side_by_side=use_sbs)
        except Exception as ex:
            print(f"[DEMO] undistort demo failed: {ex}")


def load_ost_intrinsics(ost_path: str):
    """Load K (3x3) and D (1xN) from a ROS-style ost.yaml or camera_info yaml."""
    import yaml
    with open(ost_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # camera_matrix
    if "camera_matrix" in data and "data" in data["camera_matrix"]:
        k_list = data["camera_matrix"]["data"]
    elif "K" in data:
        k_list = data["K"]
    else:
        raise ValueError("Cannot find camera_matrix / K in yaml")

    K = np.array(k_list, dtype=np.float64).reshape(3, 3)

    # distortion
    if "distortion_coefficients" in data and "data" in data["distortion_coefficients"]:
        d_list = data["distortion_coefficients"]["data"]
    elif "D" in data:
        d_list = data["D"]
    elif "distortion" in data:
        d_list = data["distortion"]
    else:
        d_list = []

    D = np.array(d_list, dtype=np.float64).reshape(1, -1)
    w = int(data.get("image_width", 0))
    h = int(data.get("image_height", 0))
    return K, D, (w, h)


def demo_undistort(input_video: str, ost_path: str, output_video: str = None, num_frames: int = 60, side_by_side: bool = False):
    """Quick demo: read frames from input_video, undistort using the ost.yaml, write to output_video.

    If side_by_side=True, the output will be RAW | UNDISTORTED horizontally concatenated (useful for visual verification).
    """
    K, D, (w, h) = load_ost_intrinsics(ost_path)
    print(f"[DEMO] Loaded K from {ost_path}, image size {w}x{h}")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("[DEMO] Cannot open input video")
        return

    # Compute optimal new camera matrix (alpha=0 keeps valid area)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0)

    # Prepare maps once
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_32FC1)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        if side_by_side:
            side = cv2.hconcat([frame, undist])
            cv2.putText(side, "RAW", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(side, "UNDIST", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            frames.append(side)
        else:
            frames.append(undist)

    cap.release()

    if not frames:
        print("[DEMO] No frames read")
        return

    out_w = w * 2 if side_by_side else w
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(output_video, fourcc, 30.0, (out_w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"[DEMO] Wrote {'side-by-side ' if side_by_side else ''}preview: {output_video} ({len(frames)} frames)")
    else:
        for i, f in enumerate(frames[:3]):
            cv2.imshow(("SideBySide" if side_by_side else "Undistorted") + f" {i}", f)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
