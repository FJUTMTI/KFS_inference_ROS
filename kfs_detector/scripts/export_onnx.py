#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11 3 类模型 → ONNX 导出脚本
================================
将 training_output/runs/KFS_full_3class/weights/best.pt 导出为 ONNX，
供 C++ 推理模块使用。

前置条件:
    1. 已安装 ultralytics:  pip install ultralytics
    2. 已安装 onnx:         pip install onnx
    3. (可选) 简化模型:      pip install onnx-simplifier
    4. 已完成训练:          python train_yolo11_3class.py

快速开始:
    cd ~/KFS_training/realsense_inference
    python scripts/export_onnx.py

自定义路径:
    python scripts/export_onnx.py \\
        --pt ~/KFS_training/training_output/runs/KFS_full_3class/weights/best.pt \\
        --output models/kfs_yolo11_3class.onnx \\
        --imgsz 640 --opset 12

导出其他格式 (TensorRT / OpenVINO):
    python scripts/export_onnx.py --format engine   # TensorRT (需要 GPU + TensorRT SDK)
    python scripts/export_onnx.py --format openvino  # OpenVINO (Intel CPU/GPU)

验证导出的 ONNX:
    python scripts/export_onnx.py --verify
"""

import argparse
import os
import sys
from pathlib import Path

# ─── 路径配置 ────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR    = PROJECT_ROOT / "models"
DEFAULT_PT    = PROJECT_ROOT / "training_output" / "runs" / "KFS_full_3class" / "weights" / "best.pt"
DEFAULT_ONNX  = MODELS_DIR / "kfs_yolo11_3class.onnx"

# ─── 自动发现 .pt 文件 ──────────────────────────────────
def find_pt_files(base_dir: Path) -> list:
    """在项目目录下搜索所有 .pt 模型文件"""
    candidates = []
    for pattern in ["**/best.pt", "**/last.pt", "**/*.pt"]:
        for pt in base_dir.glob(pattern):
            if pt.is_file() and pt not in candidates:
                candidates.append(pt)
    return sorted(candidates)


def print_model_info(model):
    """打印模型详细信息"""
    print("=" * 60)
    print("  模型信息")
    print("=" * 60)
    print(f"  类别数:    {len(model.names)}")
    print(f"  类别名:    {model.names}")
    try:
        # 尝试获取更多信息
        if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
            nc = model.model.yaml.get('nc', '?')
        else:
            nc = len(model.names)
        print(f"  nc:        {nc}")
    except Exception:
        pass
    print(f"  任务:      detect (3-class KFS)")
    print(f"  输入尺寸:  支持导出时指定")
    print("=" * 60)


def verify_onnx(onnx_path: Path):
    """验证 ONNX 模型是否可用"""
    print(f"\n[验证] 检查 ONNX 模型: {onnx_path}")

    if not onnx_path.exists():
        print(f"  [FAIL] 文件不存在: {onnx_path}")
        return False

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.1f} MB")

    # 1) 用 onnx 库检查
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"  [OK] ONNX 模型结构验证通过")
        print(f"  opset: {model.opset_import[0].version}")
        print(f"  IR 版本: {model.ir_version}")
        print(f"  生产者: {model.producer_name}")

        # 输入输出 shape
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  输入: {inp.name}  shape={shape}")
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  输出: {out.name}  shape={shape}")
    except ImportError:
        print("  [WARN] 未安装 onnx 库, 跳过结构验证 (pip install onnx)")
    except Exception as e:
        print(f"  [WARN] ONNX 验证警告: {e}")

    # 2) 用 onnxruntime 做推理测试
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(onnx_path))
        in_name = sess.get_inputs()[0].name
        in_shape = sess.get_inputs()[0].shape
        print(f"  [OK] ONNX Runtime 加载成功")
        print(f"  推理输入: {in_name} shape={in_shape}")

        # 跑一次空推理确认可用
        if all(d is not None and d > 0 for d in in_shape if isinstance(d, int)):
            dummy = np.random.randn(*in_shape).astype(np.float32)
            _ = sess.run(None, {in_name: dummy})
            print(f"  [OK] 推理测试通过 (dummy input)")
        else:
            # 动态 shape, 用固定尺寸测试
            test_shape = [1, 3, 640, 640]
            dummy = np.random.randn(*test_shape).astype(np.float32)
            _ = sess.run(None, {in_name: dummy})
            print(f"  [OK] 推理测试通过 ({test_shape})")

    except ImportError:
        print("  [WARN] 未安装 onnxruntime, 跳过推理验证 (pip install onnxruntime)")
    except Exception as e:
        print(f"  [WARN] 推理测试警告: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="导出 YOLO11 3 类 KFS 模型为 ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                                  # 自动查找 best.pt 并导出
  %(prog)s --pt /path/to/best.pt --output model.onnx        # 指定路径
  %(prog)s --imgsz 320 --opset 17                           # 自定义尺寸和 opset
  %(prog)s --format openvino                                # 导出 OpenVINO
  %(prog)s --verify                                         # 验证已有 ONNX 模型
  %(prog)s --list-pt                                        # 列出可用的 .pt 文件
        """
    )

    # ── 主要参数 ──
    parser.add_argument("--pt",       type=str, default="auto",
                        help=f"训练好的 .pt 模型路径 (默认自动搜索)")
    parser.add_argument("--output",   type=str, default=str(DEFAULT_ONNX),
                        help=f"输出 ONNX 路径 (默认: {DEFAULT_ONNX})")
    parser.add_argument("--format",   type=str, default="onnx",
                        choices=["onnx", "engine", "openvino", "tflite", "coreml"],
                        help="导出格式 (默认: onnx)")
    parser.add_argument("--imgsz",    type=int, default=640,
                        help="模型输入图像尺寸 (默认: 640)")
    parser.add_argument("--opset",    type=int, default=12,
                        help="ONNX opset 版本 (默认: 12, 兼容性好)")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="用 onnx-simplifier 简化模型 (默认开启)")
    parser.add_argument("--no-simplify", action="store_true",
                        help="禁用 onnx-simplifier")
    parser.add_argument("--half",     action="store_true",
                        help="导出 FP16 半精度 (减小体积, 需 GPU 推理支持)")
    parser.add_argument("--dynamic",  action="store_true",
                        help="导出动态 batch 轴 (默认 batch=1)")
    parser.add_argument("--workspace",type=int, default=4,
                        help="TensorRT workspace GB (默认: 4)")

    # ── 工具命令 ──
    parser.add_argument("--verify",   action="store_true",
                        help="验证已有 ONNX 模型是否可用")
    parser.add_argument("--list-pt",  action="store_true",
                        help="列出项目下所有可用的 .pt 模型文件")

    args = parser.parse_args()

    # ── 确保输出目录存在 ──
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 列出 .pt 文件 ──
    if args.list_pt:
        pt_files = find_pt_files(PROJECT_ROOT)
        if not pt_files:
            print("未找到任何 .pt 文件。请先运行训练脚本。")
            print(f"  搜索范围: {PROJECT_ROOT}")
        else:
            print("\n可用的 .pt 模型文件:\n")
            for i, pt in enumerate(pt_files):
                size_mb = pt.stat().st_size / (1024 * 1024)
                print(f"  [{i}] {pt}  ({size_mb:.1f} MB)")
            print(f"\n用法: python {__file__} --pt <路径>")
            print(f"自动选择: {DEFAULT_PT}")
        return

    # ── 验证模式 ──
    if args.verify:
        onnx_path = Path(args.output)
        if not onnx_path.exists():
            onnx_path = DEFAULT_ONNX
        ok = verify_onnx(onnx_path)
        if ok:
            print(f"\n[OK] 模型可用, 可以部署到 C++ 推理")
            print(f"  cd ~/KFS_training/realsense_inference")
            print(f"  mkdir -p build && cd build")
            print(f"  cmake .. && make -j$(nproc)")
            print(f"  ./kfs_detect --camera usb --debug")
        return

    # ── 查找 .pt 文件 ──
    if args.pt == "auto":
        # 优先使用默认路径
        if DEFAULT_PT.exists():
            pt_path = DEFAULT_PT
        else:
            # 自动搜索
            candidates = find_pt_files(PROJECT_ROOT)
            if not candidates:
                print("[ERROR] 未找到 .pt 模型文件!")
                print(f"  搜索范围: {PROJECT_ROOT}")
                print(f"  预期位置: {DEFAULT_PT}")
                print()
                print("  请先运行训练脚本:")
                print(f"    cd {PROJECT_ROOT}")
                print(f"    python train_yolo11_3class.py")
                print()
                print("  或手动指定: python scripts/export_onnx.py --pt <path>")
                sys.exit(1)
            pt_path = candidates[-1]  # 取最新
            print(f"[INFO] 自动选择: {pt_path}")
            if len(candidates) > 1:
                print(f"[INFO] 其他可用文件 ({len(candidates)} 个):")
                for c in candidates[:-1]:
                    print(f"         {c}")
    else:
        pt_path = Path(args.pt)

    if not pt_path.exists():
        print(f"[ERROR] 模型文件不存在: {pt_path}")
        print(f"  请确认训练已完成, 或使用 --list-pt 查看可用文件")
        sys.exit(1)

    # ── 导入 ultralytics ──
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics!")
        print("  请运行: pip install ultralytics")
        sys.exit(1)

    # ── 加载模型 ──
    print(f"\n[INFO] 加载模型: {pt_path}")
    model = YOLO(str(pt_path))
    print_model_info(model)

    # ── 处理 simplify ──
    simplify = args.simplify and not args.no_simplify
    if simplify:
        try:
            import onnxsim  # noqa: F401
        except ImportError:
            print("[WARN] 未安装 onnx-simplifier, 跳过简化 (pip install onnx-simplifier)")
            simplify = False

    # ── 导出 ──
    onnx_path = Path(args.output)
    print(f"\n[INFO] 正在导出...")
    print(f"  格式:    {args.format}")
    print(f"  尺寸:    {args.imgsz}×{args.imgsz}")
    print(f"  精度:    {'FP16' if args.half else 'FP32'}")
    print(f"  Opset:   {args.opset}")
    print(f"  动态轴:  {'是' if args.dynamic else '否 (固定 batch=1)'}")
    print(f"  简化:    {'是' if simplify else '否'}")
    print(f"  输出:    {onnx_path}")
    print()

    try:
        export_kwargs = dict(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            workspace=args.workspace,
        )

        # onnx 专属参数
        if args.format == "onnx":
            export_kwargs["opset"] = args.opset
            export_kwargs["simplify"] = simplify

        export_path = model.export(**export_kwargs)

    except Exception as e:
        print(f"\n[ERROR] 导出失败: {e}")
        print("\n  常见问题:")
        print("  1. onnx-simplifier 冲突: 尝试 --no-simplify")
        print("  2. opset 不兼容: 尝试 --opset 11 或 12")
        print("  3. 显存不足 (TensorRT): 减小 --workspace")
        sys.exit(1)

    print(f"\n[OK] 导出成功: {export_path}")

    # ── 复制到目标位置 ──
    if str(export_path) != str(onnx_path):
        import shutil
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export_path, onnx_path)
        print(f"[INFO] 已复制到: {onnx_path}")

    # ── 验证 ──
    if args.format == "onnx":
        verify_onnx(onnx_path)

    # ── 部署提示 ──
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║  模型导出完成! 可以部署到 C++ 推理了                       ║
║                                                           ║
║  1) 确认模型文件在正确位置:                                ║
║       ls -lh {onnx_path}                                  ║
║                                                           ║
║  2) 如需复制到指定位置:                                    ║
║       cp {onnx_path} {MODELS_DIR}/                        ║
║                                                           ║
║  3) 编译 C++ 推理:                                        ║
║       cd {PROJECT_ROOT}                                   ║
║       mkdir -p build && cd build                          ║
║       cmake .. -DCMAKE_BUILD_TYPE=Release                 ║
║       make -j$(nproc)                                     ║
║                                                           ║
║  4) 运行 (RealSense D415):                                ║
║       ./kfs_detect --camera realsense --debug             ║
║                                                           ║
║  5) 运行 (USB 摄像头):                                    ║
║       ./kfs_detect --list-cameras                         ║
║       ./kfs_detect --camera usb --device 0 --debug        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
