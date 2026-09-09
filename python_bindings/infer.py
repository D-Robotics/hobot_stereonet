import os
import cv2
import sys
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "build")
sys.path.append(BUILD_DIR)

import dstereonet


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ================================ file utils (mirror FileUtils in file_utils.cpp) ================================

def is_image_file(filename):
    return os.path.splitext(filename)[1] in IMAGE_EXTS


def find_images(folder):
    """List all image files in the folder, sorted by filename (FileUtils::find_images)."""
    if not os.path.isdir(folder):
        return []
    images = []
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and is_image_file(f):
            images.append(os.path.abspath(p))
    return images


def find_pairs(folder):
    """Find left/right image pairs by replacing 'left' with 'right' in the filename
    (FileUtils::find_pairs)."""
    if not os.path.isdir(folder):
        return []
    pairs = []
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if not os.path.isfile(p):
            continue
        if is_image_file(f) and "left" in f:
            right_path = os.path.join(folder, f.replace("left", "right", 1))
            if os.path.exists(right_path):
                pairs.append((os.path.abspath(p), os.path.abspath(right_path)))
    pairs.sort(key=lambda x: os.path.basename(x[0]))
    return pairs


def _only_images(folder):
    if not os.path.isdir(folder):
        return False
    for entry in os.listdir(folder):
        p = os.path.join(folder, entry)
        if os.path.isdir(p):
            return False
        if os.path.isfile(p) and not is_image_file(entry):
            return False
    return True


def has_left_right_dirs(folder):
    """True if the folder has left/ and right/ subdirectories containing only images."""
    return _only_images(os.path.join(folder, "left")) and _only_images(os.path.join(folder, "right"))


def find_left_right_pairs(folder):
    """Match images between left/ and right/ subdirectories by stem (infer.cpp findLeftRightPairs)."""
    left_dir = os.path.join(folder, "left")
    right_dir = os.path.join(folder, "right")
    if not (os.path.isdir(left_dir) and os.path.isdir(right_dir)):
        return []

    left_map = {}
    for f in os.listdir(left_dir):
        p = os.path.join(left_dir, f)
        if os.path.isfile(p) and is_image_file(f):
            left_map[os.path.splitext(f)[0]] = os.path.abspath(p)

    pairs = []
    for f in os.listdir(right_dir):
        p = os.path.join(right_dir, f)
        if os.path.isfile(p) and is_image_file(f):
            stem = os.path.splitext(f)[0]
            if stem in left_map:
                pairs.append((left_map[stem], os.path.abspath(p)))
    pairs.sort(key=lambda x: os.path.basename(x[0]))
    return pairs


# ================================ data helpers ================================

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


def read_camera_intrinsic(file_path):
    """Read camera intrinsic from a file, matching readCameraIntrinsicFromFile in infer.cpp.

    Supported formats (comments starting with '#' and blank lines are ignored):
      - 5 values : fx fy cx cy baseline
      - 10 values: 3x3 K (fx 0 cx / 0 fy cy / 0 0 1) followed by baseline
    Returns a dict {fx, fy, cx, cy, baseline, doffs}, or None on failure.
    """
    values = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for tok in line.split():
                    values.append(float(tok))
    except (OSError, ValueError):
        return None

    if len(values) == 5:
        return {"fx": values[0], "fy": values[1], "cx": values[2], "cy": values[3],
                "baseline": values[4], "doffs": 0.0}
    if len(values) == 10:
        return {"fx": values[0], "cx": values[2], "fy": values[4], "cy": values[5],
                "baseline": values[9], "doffs": 0.0}
    return None


def build_camera_intrinsic(intr):
    """Build a dstereonet.CameraIntrinsic from a parsed intrinsic dict.

    A None dict yields a default (all-zero) intrinsic, matching the C++ behavior of
    an uninitialized CameraIntrinsic (is_valid() == false).
    """
    cam = dstereonet.CameraIntrinsic()
    if intr is not None:
        cam.fx = float(intr["fx"])
        cam.fy = float(intr["fy"])
        cam.cx = float(intr["cx"])
        cam.cy = float(intr["cy"])
        cam.baseline = float(intr["baseline"])
        cam.doffs = float(intr.get("doffs", 0.0))
        cam.rectify_model = intr.get("rectify_model", "RECTIFY_PERSPECTIVE")
    return cam


def save_camera_intrinsic(result_dir, cam):
    """Write camera_intrinsic.txt and K.txt, matching infer.cpp saveCameraIntrinsic."""
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "camera_intrinsic.txt"), "w") as f:
        f.write("# fx fy cx cy baseline(m)\n")
        f.write(f"{cam.fx} {cam.fy} {cam.cx} {cam.cy} {cam.baseline}\n")
    with open(os.path.join(result_dir, "K.txt"), "w") as f:
        f.write(f"{cam.fx} 0.0 {cam.cx} 0.0 {cam.fy} {cam.cy} 0.0 0.0 1.0\n")
        f.write(f"{cam.baseline}\n")


# ================================ scene processing ================================

def process_one_scene_dir(scene_dir, root_dir, result_root, net, model_w, model_h, uncertainty_th):
    img_pairs = find_pairs(scene_dir)
    single_img_paths = []
    use_vert_split = False
    use_left_right_dirs = False

    if not img_pairs:
        if has_left_right_dirs(scene_dir):
            img_pairs = find_left_right_pairs(scene_dir)
            if not img_pairs:
                print(f"[WARN] no matching image pairs found in left/right subdirs: {scene_dir}")
                return False
            use_left_right_dirs = True
            print(f"[INFO] found {len(img_pairs)} image pairs via left/right subdirs")
        else:
            single_img_paths = find_images(scene_dir)
            if not single_img_paths:
                print(f"[WARN] no image pairs or images found in {scene_dir}")
                return False
            use_vert_split = True

    # result dir: <result_root>/<root_name>/<relative path>
    root_path = os.path.realpath(root_dir)
    scene_path = os.path.realpath(scene_dir)
    root_name = os.path.basename(root_path)
    rel_path = os.path.relpath(scene_path, root_path)
    if rel_path in ("", "."):
        result_dir = os.path.join(result_root, root_name)
    else:
        result_dir = os.path.join(result_root, root_name, rel_path)
    os.makedirs(result_dir, exist_ok=True)

    print(f"[INFO] ==============================================")
    print(f"[INFO] processing folder: {scene_dir}")
    print(f"[INFO] result dir: {result_dir}")

    # read intrinsic (camera_intrinsic.txt then K.txt), same as infer.cpp
    intrinsic = None
    for name in ("camera_intrinsic.txt", "K.txt"):
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            intrinsic = read_camera_intrinsic(p)
            if intrinsic is not None:
                print(f"[INFO] cam intrinsic [fx,fy,cx,cy,baseline]: "
                      f"[{intrinsic['fx']}, {intrinsic['fy']}, {intrinsic['cx']}, "
                      f"{intrinsic['cy']}, {intrinsic['baseline']}]")
                break
            print(f"[WARN] failed to parse intrinsic file: {p}")
    if intrinsic is None:
        print(f"[WARN] no intrinsic file found in {scene_dir}")

    # build work items
    work_items = []
    for (lp, rp) in img_pairs:
        work_items.append({
            "left_path": lp, "right_path": rp, "stacked_path": "",
            "prefix": os.path.splitext(os.path.basename(lp))[0],
            "vert_split": False, "left_right_dirs": use_left_right_dirs,
        })
    if use_vert_split:
        for sp in single_img_paths:
            work_items.append({
                "left_path": "", "right_path": "", "stacked_path": sp,
                "prefix": os.path.splitext(os.path.basename(sp))[0],
                "vert_split": True, "left_right_dirs": False,
            })

    update_cam_intr = False
    cam = None
    for item in work_items:
        # read / split images
        if not item["vert_split"]:
            left_img = cv2.imread(item["left_path"], cv2.IMREAD_COLOR)
            right_img = cv2.imread(item["right_path"], cv2.IMREAD_COLOR)
            if left_img is None or right_img is None:
                print(f"[ERROR] image read failed: {item['left_path']} / {item['right_path']}")
                continue
            if item["left_right_dirs"]:
                left_img_name = "left_" + os.path.basename(item["left_path"])
                right_img_name = "right_" + os.path.basename(item["right_path"])
            else:
                left_img_name = os.path.basename(item["left_path"])
                right_img_name = os.path.basename(item["right_path"])
        else:
            print(f"[INFO] processing vertically stacked image: {item['stacked_path']}")
            stacked_img = cv2.imread(item["stacked_path"], cv2.IMREAD_COLOR)
            if stacked_img is None:
                print(f"[ERROR] image read failed: {item['stacked_path']}")
                continue
            if stacked_img.shape[0] % 2 != 0:
                print(f"[ERROR] stacked image height is odd, cannot split: {item['stacked_path']}")
                continue
            half_h = stacked_img.shape[0] // 2
            left_img = stacked_img[0:half_h, :].copy()
            right_img = stacked_img[half_h:, :].copy()
            stacked_name = os.path.basename(item["stacked_path"])
            left_img_name = "left_" + stacked_name
            right_img_name = "right_" + stacked_name

        # resize
        if left_img.shape[1] != model_w or left_img.shape[0] != model_h:
            left_img_resize = cv2.resize(left_img, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
            right_img_resize = cv2.resize(right_img, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
            # scale intrinsic once per scene (same as infer.cpp)
            if not update_cam_intr and intrinsic is not None:
                intrinsic["fx"] *= model_w / float(left_img.shape[1])
                intrinsic["cx"] *= model_w / float(left_img.shape[1])
                intrinsic["fy"] *= model_h / float(left_img.shape[0])
                intrinsic["cy"] *= model_h / float(left_img.shape[0])
                update_cam_intr = True
        else:
            left_img_resize = left_img
            right_img_resize = right_img

        # build the CameraIntrinsic once (after the one-time resize scale above)
        if cam is None:
            cam = build_camera_intrinsic(intrinsic)
            if cam.is_valid():
                save_camera_intrinsic(result_dir, cam)

        # convert to nv12
        left_nv12 = np.ascontiguousarray(bgr_to_nv12_opencv(left_img_resize), dtype=np.uint8)
        right_nv12 = np.ascontiguousarray(bgr_to_nv12_opencv(right_img_resize), dtype=np.uint8)

        expected_size = model_w * model_h * 3 // 2
        if left_nv12.size != expected_size or right_nv12.size != expected_size:
            print(f"[ERROR] NV12 size mismatch: {left_nv12.size} / {right_nv12.size} != {expected_size}")
            continue

        # infer
        disp, uncert = net.forward_sync(left_nv12, right_nv12, uncertainty_th)
        disp = np.asarray(disp, dtype=np.float32)

        # depth (only when intrinsic is valid, same as infer.cpp)
        depth_mm = None
        if cam.is_valid():
            depth_mm = dstereonet.StereonetProcess.perspective_disparity_to_depth(disp, cam)

        # epipolar alignment check (always, same as infer.cpp)
        epipolar_visual = dstereonet.check_epipolar_alignment(left_img_resize, right_img_resize, cam)

        # save (mirror infer.cpp)
        prefix = item["prefix"]
        cv2.imwrite(os.path.join(result_dir, left_img_name), left_img_resize)
        cv2.imwrite(os.path.join(result_dir, right_img_name), right_img_resize)
        write_pfm(os.path.join(result_dir, f"disp_{prefix}.pfm"), disp)
        if uncert is not None:
            write_pfm(os.path.join(result_dir, f"uncert_{prefix}.pfm"),
                      np.asarray(uncert, dtype=np.float32))
        if epipolar_visual is not None:
            cv2.imwrite(os.path.join(result_dir, f"epipolar_visual_{prefix}.png"), epipolar_visual)

        visual_disp = dstereonet.StereonetProcess.render_disp_or_depth(disp)
        cv2.imwrite(os.path.join(result_dir, f"visual_disp_{prefix}.png"), visual_disp)
        visual_disp_sf = dstereonet.StereonetProcess.render_disp_or_depth(
            disp, 0.0, 192.0, 0.0, 10000.0, True, 100, 2.0, 8)
        cv2.imwrite(os.path.join(result_dir, f"visual_disp_sf_{prefix}.png"), visual_disp_sf)

        if cam.is_valid():
            cv2.imwrite(os.path.join(result_dir, f"depth_{prefix}.png"), depth_mm)
            visual = dstereonet.StereonetProcess.convert_visual_img(left_img_resize, disp, depth_mm, cam)
            cv2.imwrite(os.path.join(result_dir, f"visual_{prefix}.png"), visual)

            pointcloud = dstereonet.StereonetProcess.depth_to_pointcloud_rgb(depth_mm, left_img_resize, cam)
            dstereonet.StereonetProcess.dump_pcd_file_rgb(
                os.path.join(result_dir, f"pointcloud_{prefix}.pcd"), pointcloud)

    return True


# ================================ main ================================

def main():
    parser = argparse.ArgumentParser(
        description="Standalone StereoNet inference (folder input, same usage as ./infer)")
    parser.add_argument("model_path", nargs="?", default="./model/DStereoV2.4_int16.bin",
                        help="path to stereo model (.bin), default ./model/DStereoV2.4_int16.bin")
    parser.add_argument("local_img_dir", nargs="?", default="./img",
                        help="path to local image directory, default ./img")
    parser.add_argument("uncertainty_th", nargs="?", type=float, default=-0.10,
                        help="uncertainty threshold, default -0.10")
    parser.add_argument("--out_dir", default="./result", help="result root directory, default ./result")
    parser.add_argument("--post_version", default="auto")
    parser.add_argument("--max_memory_count", type=int, default=5)
    args = parser.parse_args()

    model_path = args.model_path
    local_img_dir = args.local_img_dir
    uncertainty_th = args.uncertainty_th

    if not os.path.exists(model_path):
        print(f"[ERROR] model file not exist: {model_path}")
        return -1
    if not os.path.isdir(local_img_dir):
        print(f"[ERROR] local image directory not exist or not directory: {local_img_dir}")
        return -1

    # init StereoNetProcess
    net = dstereonet.StereonetProcess()
    net.init(model_path, args.post_version, args.max_memory_count)
    model_w, model_h = net.get_model_input_size()
    print(f"[INFO] model input size: {model_w} x {model_h}")

    result_root = args.out_dir
    os.makedirs(result_root, exist_ok=True)

    root = os.path.abspath(local_img_dir)
    processed_dir_count = 0

    # 1. process the root directory itself
    if find_pairs(root) or find_images(root) or has_left_right_dirs(root):
        if process_one_scene_dir(root, root, result_root, net, model_w, model_h, uncertainty_th):
            processed_dir_count += 1

    # 2. recursively process all subdirectories
    for sub_dir, _, _ in os.walk(root):
        if sub_dir == root:
            continue
        # skip left/right subdirectories that belong to a left/right-dirs parent
        parent = os.path.dirname(sub_dir)
        if has_left_right_dirs(parent):
            if os.path.basename(sub_dir) in ("left", "right"):
                continue
        if not (find_pairs(sub_dir) or find_images(sub_dir) or has_left_right_dirs(sub_dir)):
            continue
        if process_one_scene_dir(sub_dir, root, result_root, net, model_w, model_h, uncertainty_th):
            processed_dir_count += 1

    if processed_dir_count == 0:
        print(f"[WARN] no valid scene directory found under: {local_img_dir}")

    print(f"[INFO] ==============================================")
    print(f"[INFO] done, processed dir count: {processed_dir_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
