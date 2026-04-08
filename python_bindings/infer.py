import os
import cv2
import sys
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "build")
sys.path.append(BUILD_DIR)

import dstereonet


def write_pfm(path, image, scale=1.0):
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    if image.ndim == 2:
        color = False
    elif image.ndim == 3 and image.shape[2] == 3:
        color = True
    else:
        raise ValueError("PFM image must have shape HxW or HxWx3")

    image = np.flipud(image)

    with open(path, "wb") as f:
        f.write(b"PF\n" if color else b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode())

        endian = image.dtype.byteorder
        if endian == "<" or (endian == "=" and sys.byteorder == "little"):
            scale = -scale
        f.write(f"{scale}\n".encode())

        image.tofile(f)


def bgr_to_nv12_opencv(bgr):
    h, w = bgr.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("NV12 requires even width and height")

    yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    yuv_i420 = yuv_i420.reshape(-1)

    y_size = w * h
    uv_size = y_size // 4

    y = yuv_i420[:y_size]
    u = yuv_i420[y_size:y_size + uv_size]
    v = yuv_i420[y_size + uv_size:y_size + uv_size * 2]

    # I420: YYYY... UU... VV...
    # NV12: YYYY... UVUVUV...
    uv = np.empty((uv_size * 2,), dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v

    nv12 = np.concatenate([y, uv])
    return nv12


def disparity_to_depth_mm(disp, fx, baseline_m, doffs=0.0, invalid_depth_mm=0):
    disp = disp.astype(np.float32)
    denom = disp + float(doffs)

    depth_m = np.zeros_like(disp, dtype=np.float32)
    valid = denom > 1e-6
    depth_m[valid] = (fx * baseline_m) / denom[valid]

    depth_mm = np.full_like(disp, invalid_depth_mm, dtype=np.uint16)

    valid_mm = valid & np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < 65.535)
    depth_mm[valid_mm] = np.round(depth_m[valid_mm] * 1000.0).astype(np.uint16)

    return depth_mm


def render_disp_for_vis(disp, max_disp=192.0):
    disp_vis = np.clip(disp, 0, max_disp)
    disp_vis = (disp_vis / max_disp * 255.0).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    return disp_vis


def load_and_prepare_bgr(img_path, target_w, target_h):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    if img.shape[1] != target_w or img.shape[0] != target_h:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="StereoNet model path")
    parser.add_argument("--left", required=True, help="left png path")
    parser.add_argument("--right", required=True, help="right png path")
    parser.add_argument("--out_dir", default="./output", help="output directory")
    parser.add_argument("--uncertainty_th", type=float, default=0.3)

    parser.add_argument("--fx", type=float, required=True, help="focal length fx in pixels")
    parser.add_argument("--baseline", type=float, required=True, help="baseline in meters")
    parser.add_argument("--doffs", type=float, default=0.0, help="disparity offset")

    parser.add_argument("--post_version", default="auto")
    parser.add_argument("--max_memory_count", type=int, default=5)
    parser.add_argument("--save_vis", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    net = dstereonet.StereonetProcess()
    net.init(args.model, args.post_version, args.max_memory_count)

    model_w, model_h = net.get_model_input_size()
    print(f"[INFO] model input size: {model_w} x {model_h}")

    left_bgr = load_and_prepare_bgr(args.left, model_w, model_h)
    right_bgr = load_and_prepare_bgr(args.right, model_w, model_h)

    left_nv12 = bgr_to_nv12_opencv(left_bgr)
    right_nv12 = bgr_to_nv12_opencv(right_bgr)

    left_nv12 = np.ascontiguousarray(left_nv12, dtype=np.uint8)
    right_nv12 = np.ascontiguousarray(right_nv12, dtype=np.uint8)

    expected_size = model_w * model_h * 3 // 2
    if left_nv12.size != expected_size:
        raise RuntimeError(f"left NV12 size mismatch: {left_nv12.size} != {expected_size}")
    if right_nv12.size != expected_size:
        raise RuntimeError(f"right NV12 size mismatch: {right_nv12.size} != {expected_size}")

    disp, uncert = net.forward_sync(left_nv12, right_nv12, args.uncertainty_th)
    if uncert is None:
      print("uncert is empty")

    disp = np.asarray(disp, dtype=np.float32)
    #uncert = np.asarray(uncert, dtype=np.float32)

    print(f"[INFO] disp shape: {disp.shape}, dtype: {disp.dtype}")
    #print(f"[INFO] uncert shape: {uncert.shape}, dtype: {uncert.dtype}")

    disp_pfm_path = os.path.join(args.out_dir, "disp.pfm")
    write_pfm(disp_pfm_path, disp)
    print(f"[INFO] saved disparity pfm: {disp_pfm_path}")

    depth_mm = disparity_to_depth_mm(
        disp=disp,
        fx=args.fx,
        baseline_m=args.baseline,
        doffs=args.doffs
    )

    depth_png_path = os.path.join(args.out_dir, "depth_mm.png")
    ok = cv2.imwrite(depth_png_path, depth_mm)
    if not ok:
        raise RuntimeError(f"Failed to save depth png: {depth_png_path}")
    print(f"[INFO] saved depth png(uint16 mm): {depth_png_path}")

    #uncert_npy_path = os.path.join(args.out_dir, "uncert.npy")
    #np.save(uncert_npy_path, uncert)
    #print(f"[INFO] saved uncertainty npy: {uncert_npy_path}")

    if args.save_vis:
        disp_vis = render_disp_for_vis(disp, max_disp=192.0)
        cv2.imwrite(os.path.join(args.out_dir, "disp_vis.png"), disp_vis)

        depth_vis = depth_mm.astype(np.float32)
        valid = depth_vis > 0
        if np.any(valid):
            vmin = np.percentile(depth_vis[valid], 2)
            vmax = np.percentile(depth_vis[valid], 98)
            depth_vis = np.clip(depth_vis, vmin, vmax)
            depth_vis = ((depth_vis - vmin) / max(vmax - vmin, 1e-6) * 255.0).astype(np.uint8)
            depth_vis[~valid] = 0
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.out_dir, "depth_vis.png"), depth_vis)

        print(f"[INFO] saved visualization images to: {args.out_dir}")


if __name__ == "__main__":
    main()