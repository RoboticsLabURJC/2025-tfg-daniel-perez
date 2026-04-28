import os
import cv2
import numpy as np
import json


# =========================================================
# CONFIGURACIÓN
# =========================================================

VALID_CLASSES = [1,2,3,4,5,6,8,9,10,11]

ONTOLOGY_REMAP = {i: i for i in range(12)}

CLASS_COLOR_MAP = {
    0: (0, 0, 0),        # void
    1: (0, 255, 0),      # vegetation
    2: (128, 64, 32),    # terrain
    3: (0, 0, 128),      # sky
    4: (255, 0, 0),      # construction
    5: (255, 0, 255),    # vehicle
    6: (128, 128, 128),  # road
    7: (0, 255, 255),    # object
    8: (255, 255, 0),    # sign
    9: (255, 128, 0),    # human
    10: (0, 0, 255),     # water
    11: (128, 0, 128),   # animal
}

def remap_labels(labels, remap_dict):
    out = np.zeros_like(labels, dtype=np.uint8)
    for src, dst in remap_dict.items():
        out[labels == src] = dst
    return out

def load_calib(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    R0 = np.array(data["rmat"], dtype=np.float64)
    t0 = np.array(data["tvec"], dtype=np.float64).reshape(3)
    K = np.array(data["cam_mat"], dtype=np.float64)
    image_size = tuple(data["image_size"])   # [W, H]

    return R0, t0, K, image_size

def load_rgb_seg(png_path):
    """
    Carga la segmentación 2D como imagen de clases (H, W).
    """
    seg = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        raise FileNotFoundError(f"No se pudo leer: {png_path}")

    if seg.ndim == 3:
        # Si viniera en color, usamos el primer canal
        seg = seg[:, :, 0]

    return seg.astype(np.int32)

def load_lidar_points(bin_path):
    """
    Carga puntos LiDAR tipo float32.
    Formato esperado: x, y, z, intensidad
    """
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("El .bin no tiene múltiplo de 4 float32")

    pts = raw.reshape(-1, 4)
    xyz = pts[:, :3]
    intensity = pts[:, 3]
    return xyz, intensity

def load_lidar_labels(label_path):
    """
    Carga una etiqueta por punto LiDAR.
    """
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels.astype(np.int32)

def project_points(points_xyz, R, t, K):
    """
    Proyecta puntos 3D LiDAR sobre la imagen.
    Retorna:
      uv: (N, 2)
      z_cam: (N,)
    """
    pts_cam = (R @ points_xyz.T).T + t
    z_cam = pts_cam[:, 2].copy()

    # evitar división por cero
    eps = 1e-12
    valid_z = np.abs(z_cam) > eps

    uv = np.full((points_xyz.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid_z):
        pts_valid = pts_cam[valid_z]
        proj = (K @ pts_valid.T).T
        uv_valid = proj[:, :2] / proj[:, 2:3]
        uv[valid_z] = uv_valid

    return uv, z_cam

# =========================================================
# 1) Rotación + escalado
# =========================================================
def rotate_scale_patch(patch_bgr, angle_deg, scale=1.0):
    h, w = patch_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))

    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    warped = cv2.warpAffine(
        patch_bgr,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return warped, M

def compute_class_weights_from_patch(patch_labels, class_ids):
    """
    Calcula pesos automáticamente según el área de cada clase
    en la región válida del patch.
    Clases pequeñas -> peso alto.
    Clases grandes -> peso bajo.
    """
    weights = {}

    valid_mask = (patch_labels > 0)
    total_valid = np.count_nonzero(valid_mask)

    if total_valid == 0:
        return {c: 0.0 for c in class_ids}

    for c in class_ids:
        area = np.count_nonzero(patch_labels == c)

        if area == 0:
            weights[c] = 0.0
            continue

        ratio = area / total_valid
        weight = 1.0 / np.sqrt(ratio + 1e-6)
        weights[c] = weight

    max_w = max(weights.values()) if weights else 1.0
    if max_w > 0:
        for c in weights:
            weights[c] /= max_w

    return weights


# =========================================================
# 2) Segmentación multiclase
# =========================================================
def multiclass_mask_from_image(img_bgr):
    labels = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    color_to_class = {
        (0, 0, 0): 0,        # void
        (0, 255, 0): 1,      # vegetation
        (128, 64, 32): 2,    # terrain
        (0, 0, 128): 3,      # sky
        (255, 0, 0): 4,      # construction
        (255, 0, 255): 5,    # vehicle
        (128, 128, 128): 6,  # road
        (0, 255, 255): 7,    # object
        (255, 255, 0): 8,    # sign
        (255, 128, 0): 9,    # human
        (0, 0, 255): 10,     # water
        (128, 0, 128): 11,   # animal
    }

    for color, cls in color_to_class.items():
        mask = np.all(img_bgr == np.array(color, dtype=np.uint8), axis=2)
        labels[mask] = cls

    return labels

# =========================================================
# 3) Máscara válida
# =========================================================
def valid_region_from_labels(labels):
    return ((labels > 0).astype(np.uint8) * 255)

# =========================================================
# 4) Recorte bbox válida
# =========================================================
def crop_to_valid_region(img_or_labels, valid_mask):
    ys, xs = np.where(valid_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None, None

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    crop = img_or_labels[y0:y1, x0:x1]
    mask_crop = valid_mask[y0:y1, x0:x1]

    return crop, mask_crop, (x0, y0, x1, y1)


# =========================================================
# 5) Visualización de labels
# =========================================================
def colorize_labels(labels):
    out = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for c, color in CLASS_COLOR_MAP.items():
        out[labels == c] = color
    return out

# =========================================================
# 6) Overlay
# =========================================================
def build_overlay(big, patch, valid_mask, top_left, alpha=0.45):
    out = big.copy()

    H, W = out.shape[:2]
    h, w = patch.shape[:2]
    x, y = top_left

    x0_dst = max(0, x)
    y0_dst = max(0, y)
    x1_dst = min(W, x + w)
    y1_dst = min(H, y + h)

    if x0_dst >= x1_dst or y0_dst >= y1_dst:
        return out

    x0_src = x0_dst - x
    y0_src = y0_dst - y
    x1_src = x0_src + (x1_dst - x0_dst)
    y1_src = y0_src + (y1_dst - y0_dst)

    patch_cut = patch[y0_src:y1_src, x0_src:x1_src]
    mask_cut = valid_mask[y0_src:y1_src, x0_src:x1_src] > 0

    roi = out[y0_dst:y1_dst, x0_dst:x1_dst]
    roi[mask_cut] = (
        (1.0 - alpha) * roi[mask_cut].astype(np.float32)
        + alpha * patch_cut[mask_cut].astype(np.float32)
    ).astype(np.uint8)

    out[y0_dst:y1_dst, x0_dst:x1_dst] = roi
    return out

# =========================================================
# 7) Pegar patch
# =========================================================
def paste_with_valid_mask_clipped(background, patch, valid_mask, top_left):
    out = background.copy()

    H, W = out.shape[:2]
    h, w = patch.shape[:2]
    x, y = top_left

    x0_dst = max(0, x)
    y0_dst = max(0, y)
    x1_dst = min(W, x + w)
    y1_dst = min(H, y + h)

    if x0_dst >= x1_dst or y0_dst >= y1_dst:
        return out

    x0_src = x0_dst - x
    y0_src = y0_dst - y
    x1_src = x0_src + (x1_dst - x0_dst)
    y1_src = y0_src + (y1_dst - y0_dst)

    patch_cut = patch[y0_src:y1_src, x0_src:x1_src]
    mask_cut = valid_mask[y0_src:y1_src, x0_src:x1_src]

    roi = out[y0_dst:y1_dst, x0_dst:x1_dst]
    roi[mask_cut > 0] = patch_cut[mask_cut > 0]
    out[y0_dst:y1_dst, x0_dst:x1_dst] = roi

    return out

# =========================================================
# 8) Score grueso optimizado por clases
#    Suma ponderada de matchTemplate por clase
# =========================================================
def coarse_multiclass_score_map(big_labels, patch_labels_crop, class_ids=VALID_CLASSES, class_weights=None):
    score_sum = None
    weight_sum = 0.0

    for c in class_ids:
        big_c = ((big_labels == c).astype(np.float32)) * 255.0
        patch_c = ((patch_labels_crop == c).astype(np.float32)) * 255.0

        if np.count_nonzero(patch_c) == 0:
            continue

        try:
            res = cv2.matchTemplate(big_c, patch_c, cv2.TM_CCORR_NORMED)
        except cv2.error:
            continue

        w = float(class_weights.get(c, 1.0))
        if score_sum is None:
            score_sum = w * res
        else:
            score_sum += w * res
        weight_sum += w

    if score_sum is None or weight_sum == 0.0:
        return None

    return score_sum / weight_sum

# =========================================================
# 9) IoU multiclase ponderado y eficiente
# =========================================================
def weighted_multiclass_iou(big_labels, patch_labels, top_left, class_ids=VALID_CLASSES, class_weights=None):
    H, W = big_labels.shape[:2]
    h, w = patch_labels.shape[:2]
    x, y = top_left

    x0_dst = max(0, x)
    y0_dst = max(0, y)
    x1_dst = min(W, x + w)
    y1_dst = min(H, y + h)

    if x0_dst >= x1_dst or y0_dst >= y1_dst:
        return 0.0

    x0_src = x0_dst - x
    y0_src = y0_dst - y
    x1_src = x0_src + (x1_dst - x0_dst)
    y1_src = y0_src + (y1_dst - y0_dst)

    big_cut = big_labels[y0_dst:y1_dst, x0_dst:x1_dst]
    patch_cut = patch_labels[y0_src:y1_src, x0_src:x1_src]

    valid_patch = (patch_cut > 0)
    if np.count_nonzero(valid_patch) == 0:
        return 0.0

    total_score = 0.0
    total_weight = 0.0

    for c in class_ids:
        w_c = float(class_weights.get(c, 0.0))
        if w_c <= 0.0:
            continue

        big_c = (big_cut == c) & valid_patch
        patch_c = (patch_cut == c) & valid_patch

        union = np.logical_or(big_c, patch_c).sum()
        if union == 0:
            continue

        inter = np.logical_and(big_c, patch_c).sum()
        iou_c = inter / union

        total_score += w_c * iou_c
        total_weight += w_c

    if total_weight == 0.0:
        return 0.0

    return total_score / total_weight

# =========================================================
# 10) Búsqueda gruesa optimizada
# =========================================================
def coarse_search_multiclass_optimized(big_labels, patch_labels, out_dir):
    H, W = big_labels.shape[:2]
    best = None

    coarse_angles = list(range(-20, 21, 10))

    for angle in coarse_angles:
        patch_labels_rs, _ = rotate_scale_patch(patch_labels, angle, 1.0)
        patch_valid_rs = valid_region_from_labels(patch_labels_rs)

        patch_labels_crop, patch_valid_crop, bbox = crop_to_valid_region(patch_labels_rs, patch_valid_rs)
        if patch_labels_crop is None:
            continue

        ph, pw = patch_labels_crop.shape[:2]
        if ph <= 0 or pw <= 0 or ph > H or pw > W:
            continue

        if np.count_nonzero(patch_valid_crop) < 50:
            continue
        
        class_weights = compute_class_weights_from_patch(
                                        patch_labels_crop,
                                        VALID_CLASSES
                                    )

        score_map = coarse_multiclass_score_map(
            big_labels,
            patch_labels_crop,
            class_ids=VALID_CLASSES,
            class_weights=class_weights
        )        
        if score_map is None:
            continue

        _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
        x0, y0 = max_loc

        if best is None or max_val > best["score"]:
            best = {
                "score": float(max_val),
                "angle": float(angle),
                "top_left_crop": (int(x0), int(y0)),
                "patch_labels_rs": patch_labels_rs,
                "patch_valid_rs": patch_valid_rs,
                "patch_labels_crop": patch_labels_crop,
                "patch_valid_crop": patch_valid_crop,
                "bbox_crop_in_rs": bbox
            }

    if best is None:
        raise RuntimeError("No se encontró posición inicial en la búsqueda gruesa.")

    x0, y0 = best["top_left_crop"]
    bx0, by0, bx1, by1 = best["bbox_crop_in_rs"]

    full_x = x0 - bx0
    full_y = y0 - by0
    best["top_left_full"] = (int(full_x), int(full_y))

    rect_crop = colorize_labels(big_labels).copy()
    ph, pw = best["patch_valid_crop"].shape[:2]
    cv2.rectangle(rect_crop, (x0, y0), (x0 + pw, y0 + ph), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, "debug_rect_crop_coarse.png"), rect_crop)

    rect_full = colorize_labels(big_labels).copy()
    hr, wr = best["patch_labels_rs"].shape[:2]
    cv2.rectangle(rect_full, (full_x, full_y), (full_x + wr, full_y + hr), (0, 255, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "debug_rect_full_coarse.png"), rect_full)

    cv2.imwrite(os.path.join(out_dir, "debug_big_labels.png"), colorize_labels(big_labels))
    cv2.imwrite(os.path.join(out_dir, "debug_best_patch_labels_rs_coarse.png"), colorize_labels(best["patch_labels_rs"]))

    return best

# =========================================================
# 11) Refinado optimizado
# =========================================================
def refine_pose_multiclass_optimized(big_labels, patch_labels, init_top_left, out_dir):
    x_init, y_init = init_top_left

    def search_stage(angle_values, dx_values, dy_values, center_xy):
        cx, cy = center_xy
        best = None

        for angle in angle_values:
            patch_labels_rs, _ = rotate_scale_patch(patch_labels, angle, 1.0)

            if np.count_nonzero(patch_labels_rs > 0) < 50:
                continue

            class_weights = compute_class_weights_from_patch(
                patch_labels_rs,
                VALID_CLASSES
            )

            for dx in dx_values:
                x = cx + dx
                for dy in dy_values:
                    y = cy + dy

                    score = weighted_multiclass_iou(
                        big_labels,
                        patch_labels_rs,
                        (x, y),
                        class_ids=VALID_CLASSES,
                        class_weights=class_weights
                    )

                    if best is None or score > best["score"]:
                        best = {
                            "score": float(score),
                            "angle": float(angle),
                            "top_left_full": (int(x), int(y)),
                            "patch_labels_rs": patch_labels_rs,
                            "patch_valid_rs": valid_region_from_labels(patch_labels_rs),
                            "class_weights": class_weights,
                        }
        return best

    best1 = search_stage(
        angle_values=range(-10, 10, 10),
        dx_values=range(-12, 13, 3),
        dy_values=range(-12, 13, 3),
        center_xy=(x_init, y_init)
    )

    if best1 is None:
        raise RuntimeError("No se encontró solución en refinado etapa 1.")

    x1, y1 = best1["top_left_full"]
    a1 = int(round(best1["angle"]))

    best2 = search_stage(
        angle_values=range(a1 - 10, a1 + 11, 2),
        dx_values=range(-5, 6, 2),
        dy_values=range(-5, 6, 2),
        center_xy=(x1, y1)
    )

    if best2 is None:
        raise RuntimeError("No se encontró solución en refinado etapa 2.")

    big_bgr = colorize_labels(big_labels)
    patch_bgr_rs = colorize_labels(best2["patch_labels_rs"])

    rect_full = big_bgr.copy()
    x, y = best2["top_left_full"]
    h, w = best2["patch_labels_rs"].shape[:2]
    cv2.rectangle(rect_full, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "debug_rect_full_refined.png"), rect_full)

    overlay = build_overlay(
        big_bgr,
        patch_bgr_rs,
        best2["patch_valid_rs"],
        best2["top_left_full"]
    )
    cv2.imwrite(os.path.join(out_dir, "overlay_refined.png"), overlay)

    composite = paste_with_valid_mask_clipped(
        big_bgr,
        patch_bgr_rs,
        best2["patch_valid_rs"],
        best2["top_left_full"]
    )
    cv2.imwrite(os.path.join(out_dir, "composite_refined.png"), composite)

    cv2.imwrite(os.path.join(out_dir, "debug_best_patch_rs_refined.png"), patch_bgr_rs)
    cv2.imwrite(os.path.join(out_dir, "debug_best_patch_labels_rs_refined.png"),colorize_labels(best2["patch_labels_rs"]))
    '''    cv2.imwrite(
        os.path.join(out_dir, "debug_best_patch_valid_rs_refined.png"),
        best2["patch_valid_rs"]
    )'''
    print("Pesos usados en refined:", best2["class_weights"])

    return best2

def colorize_labels_with_black_border(labels, point_radius=1, border_thickness=1):
    H, W = labels.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    ys, xs = np.where(labels > 0)

    for y, x in zip(ys, xs):
        cls = int(labels[y, x])
        color = CLASS_COLOR_MAP.get(cls, (0, 255, 0))

        # borde negro
        cv2.circle(out, (x, y), point_radius + border_thickness, (0, 0, 0), -1)
        # relleno de clase
        cv2.circle(out, (x, y), point_radius, color, -1)

    return out

def render_lidar_patch_from_projection(
    points_xyz,
    point_labels,
    R,
    t,
    K,
    image_size
):
    H, W = image_size

    patch_labels_full = np.zeros((H, W), dtype=np.uint8)
    depth_buffer = np.full((H, W), np.inf, dtype=np.float32)

    uv, z_cam = project_points(points_xyz, R, t, K)

    valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & (z_cam > 0)

    uv_valid = uv[valid]
    z_valid = z_cam[valid]
    labels_valid = point_labels[valid]

    for (u, v), z, lbl in zip(uv_valid, z_valid, labels_valid):
        x = int(round(u))
        y = int(round(v))

        if x < 0 or x >= W or y < 0 or y >= H:
            continue

        # z-buffer: nos quedamos con el punto más cercano
        if z < depth_buffer[y, x]:
            depth_buffer[y, x] = z
            patch_labels_full[y, x] = np.uint8(lbl)

    patch_valid_full = valid_region_from_labels(patch_labels_full)
    patch_bgr_full = colorize_labels(patch_labels_full)

    return patch_labels_full, patch_bgr_full, patch_valid_full

def upscale_patch_and_mask_to_original(patch, valid_mask, top_left, original_shape, working_shape):
    H_orig, W_orig = original_shape[:2]
    H_work, W_work = working_shape[:2]

    sx = W_orig / W_work
    sy = H_orig / H_work

    patch_up = cv2.resize(
        patch,
        (int(round(patch.shape[1] * sx)), int(round(patch.shape[0] * sy))),
        interpolation=cv2.INTER_NEAREST
    )

    valid_up = cv2.resize(
        valid_mask,
        (patch_up.shape[1], patch_up.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    x, y = top_left
    top_left_up = (int(round(x * sx)), int(round(y * sy)))

    return patch_up, valid_up, top_left_up

# =========================================================
# 12) Main
# =========================================================
def main(
    calib_path,
    image_path,
    rgb_seg_path,
    points_path,
    labels_path,
    out_dir="multiclase_opt_resultados_lidar"
):
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------
    # 1) Carga de datos
    # -----------------------------------------------------
    R0, t0, K, image_size = load_calib(calib_path)
    rgb_seg = load_rgb_seg(rgb_seg_path)
    points_xyz, _ = load_lidar_points(points_path)
    lidar_labels = load_lidar_labels(labels_path)

    rgb_seg = remap_labels(rgb_seg, ONTOLOGY_REMAP)
    lidar_labels = remap_labels(lidar_labels, ONTOLOGY_REMAP)

    if len(points_xyz) != len(lidar_labels):
        raise ValueError("El número de puntos LiDAR no coincide con el número de etiquetas")

    img_original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_original is None:
        raise FileNotFoundError(f"No se pudo leer la imagen original: {image_path}")


    if len(image_size) != 2:
        raise ValueError(f"image_size inválido en calibración: {image_size}")

    W_img, H_img = int(image_size[0]), int(image_size[1])

    # Resolución real de la segmentación 2D
    seg_h, seg_w = rgb_seg.shape[:2]

    if rgb_seg.shape[:2] != (H_img, W_img):
        print("[AVISO] rgb_seg.shape y image_size no coinciden.")
        print("rgb_seg.shape:", rgb_seg.shape)
        print("image_size json [W,H]:", image_size)

        # Escalamos K para trabajar en la resolución de rgb_seg
        scale_x = seg_w / W_img
        scale_y = seg_h / H_img

        print(f"[INFO] Escalando intrínseca K con scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")

        K = K.copy()
        K[0, 0] *= scale_x   # fx
        K[0, 2] *= scale_x   # cx
        K[1, 1] *= scale_y   # fy
        K[1, 2] *= scale_y   # cy

        # A partir de aquí todo se proyecta a la resolución de rgb_seg
        image_size_hw = (seg_h, seg_w)
    else:
        image_size_hw = (H_img, W_img)

    print("rgb_seg shape         :", rgb_seg.shape)
    print("points_xyz shape      :", points_xyz.shape)
    print("lidar_labels shape    :", lidar_labels.shape)
    print("image_size json [W,H] :", image_size)

    # -----------------------------------------------------
    # 2) Big = segmentación 2D de cámara en labels
    #    y BGR solo para debug/visualización
    # -----------------------------------------------------
    big_labels = rgb_seg
    big_bgr = colorize_labels(big_labels)
    cv2.imwrite(os.path.join(out_dir, "debug_big_from_rgb_seg.png"), big_bgr)

    # -----------------------------------------------------
    # 3) Proyección inicial LiDAR -> canvas
    # -----------------------------------------------------
    patch_labels_full, patch_bgr_full, patch_valid_full = render_lidar_patch_from_projection(
        points_xyz=points_xyz,
        point_labels=lidar_labels,
        R=R0,
        t=t0,
        K=K,
        image_size=image_size_hw
    )

    cv2.imwrite(os.path.join(out_dir, "debug_patch_bgr_full_initial.png"), patch_bgr_full)
    cv2.imwrite(os.path.join(out_dir, "debug_patch_valid_full_initial.png"), patch_valid_full)

    # -----------------------------------------------------
    # 4) Recorte a región válida
    # -----------------------------------------------------
    patch_labels_crop, patch_valid_crop, bbox = crop_to_valid_region(
        patch_labels_full,
        patch_valid_full
    )

    if patch_labels_crop is None:
        raise RuntimeError("No se pudo generar un patch válido a partir de la proyección LiDAR")

    bx0, by0, bx1, by1 = bbox
    patch_bgr_crop = colorize_labels(patch_labels_crop)

    cv2.imwrite(os.path.join(out_dir, "debug_patch_bgr_crop_initial.png"), patch_bgr_crop)
    cv2.imwrite(os.path.join(out_dir, "debug_patch_valid_crop_initial.png"), patch_valid_crop)

    print("bbox patch inicial    :", bbox)
    print("patch crop shape      :", patch_labels_crop.shape)

    # -----------------------------------------------------
    # 5) Visualización inicial sobre imagen original
    # -----------------------------------------------------
    patch_bgr_crop_up, patch_valid_crop_up, top_left_crop_up = upscale_patch_and_mask_to_original(
        patch_bgr_crop,
        patch_valid_crop,
        (bx0, by0),
        original_shape=img_original.shape,
        working_shape=big_labels.shape
    )

    overlay_initial = build_overlay(
        img_original,
        patch_bgr_crop_up,
        patch_valid_crop_up,
        top_left_crop_up,
        alpha=0.45
    )
    cv2.imwrite(os.path.join(out_dir, "overlay_initial_on_rgb.png"), overlay_initial)

    # -----------------------------------------------------
    # 6) Coarse-to-fine
    # -----------------------------------------------------
    coarse = coarse_search_multiclass_optimized(
        big_labels,
        patch_labels_crop,
        out_dir
    )

    print("=== RESULTADO BÚSQUEDA GRUESA ===")
    print("score coarse         :", coarse["score"])
    print("angle coarse         :", coarse["angle"])
    print("top_left_crop        :", coarse["top_left_crop"])
    print("bbox_crop_in_rs      :", coarse["bbox_crop_in_rs"])
    print("top_left_full        :", coarse["top_left_full"])

    refined = refine_pose_multiclass_optimized(
        big_labels,
        patch_labels_crop,
        coarse["top_left_full"],
        out_dir
    )

    print("\n=== RESULTADO REFINADO ===")
    print("best weighted multiclass IoU :", refined["score"])
    print("best angle                   :", refined["angle"])
    print("best top_left                :", refined["top_left_full"])

    # -----------------------------------------------------
    # 7) Visualización final sobre imagen original
    # -----------------------------------------------------
    refined_patch_bgr = colorize_labels(refined["patch_labels_rs"])

    refined_patch_bgr_up, refined_patch_valid_up, refined_top_left_up = upscale_patch_and_mask_to_original(
        refined_patch_bgr,
        refined["patch_valid_rs"],
        refined["top_left_full"],
        original_shape=img_original.shape,
        working_shape=big_labels.shape
    )

    overlay_final_on_rgb = build_overlay(
        img_original,
        refined_patch_bgr_up,
        refined_patch_valid_up,
        refined_top_left_up,
        alpha=0.45
    )

    composite_final_on_rgb = paste_with_valid_mask_clipped(
        img_original,
        refined_patch_bgr_up,
        refined_patch_valid_up,
        refined_top_left_up
    )
    
    cv2.imwrite(os.path.join(out_dir, "overlay_refined_on_rgb.png"), overlay_final_on_rgb)

    cv2.imwrite(os.path.join(out_dir, "composite_refined_on_rgb.png"), composite_final_on_rgb)

    result = {
        "angle_deg": float(refined["angle"]),
        "tx_px": int(refined["top_left_full"][0]),
        "ty_px": int(refined["top_left_full"][1]),
        "score": float(refined["score"]),
        "class_weights": {int(k): float(v) for k, v in refined["class_weights"].items()}

    }

    with open(os.path.join(out_dir, "alignment_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return {
        "calib_path": calib_path,
        "image_path": image_path,
        "rgb_seg_path": rgb_seg_path,
        "points_path": points_path,
        "labels_path": labels_path,
        "initial_bbox": bbox,
        "coarse_score": coarse["score"],
        "coarse_angle": coarse["angle"],
        "coarse_top_left_full": coarse["top_left_full"],
        "refined_iou": refined["score"],
        "refined_angle": refined["angle"],
        "refined_top_left_full": refined["top_left_full"],
        "out_dir": out_dir,
    }


if __name__ == "__main__":
    main(
        calib_path="test_GOOSE/2022-07-22_flight-0075_1658494242083534022-calib.json",
        image_path="test_GOOSE/2022-07-22_flight-0075_1658494242083534022-image.png",
        rgb_seg_path="test_GOOSE/2022-07-22_flight-0075_1658494242083534022-pred.png",
        points_path="test_GOOSE/2022-07-22_flight-0075_1658494242083534022-points.bin",
        labels_path="test_GOOSE/2022-07-22_flight-0075_1658494242083534022-pred.label",
        out_dir="resultados_lidar_GOOSE"
    )
    
