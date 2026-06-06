import os
import cv2
import numpy as np
import json
import math
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# GLOBALES PARA WORKERS
# =========================================================
_G_BIG_LABELS = None
_G_POINTS_XYZ = None
_G_POINT_LABELS = None
_G_R0 = None
_G_T0 = None
_G_K = None
_G_IMAGE_SIZE = None
_G_CLASS_IDS = None
_G_MIN_VALID_PIXELS = None


# =========================================================
# CONFIGURACIÓN
# =========================================================

# Todas las clases excepto void
VALID_CLASSES = list(range(1, 20))

# Remapeo identidad: no se agrupa ni se elimina ninguna clase
ONTOLOGY_REMAP = {
    0: 0,    # void
    1: 1,    # dirt
    2: 2,    # grass
    3: 3,    # tree
    4: 4,    # pole
    5: 5,    # water
    6: 6,    # sky
    7: 7,    # vehicle
    8: 8,    # object
    9: 9,    # asphalt
    10: 10,  # building
    11: 11,  # log
    12: 12,  # person
    13: 13,  # fence
    14: 14,  # bush
    15: 15,  # concrete
    16: 16,  # barrier
    17: 17,  # puddle
    18: 18,  # mud
    19: 19,  # rubble
}

# Colores en formato BGR, porque cv2 usa BGR.
# Tu ontology está en RGB, por eso aquí están invertidos.
CLASS_COLOR_MAP = {
    0:  (0, 0, 0),        # void       RGB: [0, 0, 0]
    1:  (20, 64, 108),    # dirt       RGB: [108, 64, 20]
    2:  (0, 102, 0),      # grass      RGB: [0, 102, 0]
    3:  (0, 255, 0),      # tree       RGB: [0, 255, 0]
    4:  (153, 153, 0),    # pole       RGB: [0, 153, 153]
    5:  (255, 128, 0),    # water      RGB: [0, 128, 255]
    6:  (255, 0, 0),      # sky        RGB: [0, 0, 255]
    7:  (0, 255, 255),    # vehicle    RGB: [255, 255, 0]
    8:  (127, 0, 255),    # object     RGB: [255, 0, 127]
    9:  (64, 64, 64),     # asphalt    RGB: [64, 64, 64]
    10: (0, 0, 255),      # building   RGB: [255, 0, 0]
    11: (0, 0, 102),      # log        RGB: [102, 0, 0]
    12: (255, 153, 204),  # person     RGB: [204, 153, 255]
    13: (204, 0, 102),    # fence      RGB: [102, 0, 204]
    14: (204, 153, 255),  # bush       RGB: [255, 153, 204]
    15: (170, 170, 170),  # concrete   RGB: [170, 170, 170]
    16: (255, 121, 41),   # barrier    RGB: [41, 121, 255]
    17: (239, 255, 134),  # puddle     RGB: [134, 255, 239]
    18: (34, 66, 99),     # mud        RGB: [99, 66, 34]
    19: (138, 22, 110),   # rubble     RGB: [110, 22, 138]
}


def init_worker(
    big_labels,
    points_xyz,
    point_labels,
    R0,
    t0,
    K,
    image_size,
    class_ids,
    min_valid_pixels,
):
    """
    Inicializa variables globales dentro de cada worker.
    Así evitamos pasar arrays enormes en cada tarea.
    """
    global _G_BIG_LABELS, _G_POINTS_XYZ, _G_POINT_LABELS
    global _G_R0, _G_T0, _G_K, _G_IMAGE_SIZE
    global _G_CLASS_IDS, _G_MIN_VALID_PIXELS

    _G_BIG_LABELS = big_labels
    _G_POINTS_XYZ = points_xyz
    _G_POINT_LABELS = point_labels
    _G_R0 = R0
    _G_T0 = t0
    _G_K = K
    _G_IMAGE_SIZE = image_size
    _G_CLASS_IDS = class_ids
    _G_MIN_VALID_PIXELS = min_valid_pixels

def evaluate_pose_candidate_light(
    droll,
    dpitch,
    dyaw,
    dtx,
    dty,
    dtz,
):
    """
    Evalúa un candidato y devuelve SOLO lo necesario.
    No devuelve mapas renderizados para no pagar IPC innecesario.
    """
    R_cand, t_cand = compose_pose(
        R0=_G_R0,
        t0=_G_T0,
        droll_deg=droll,
        dpitch_deg=dpitch,
        dyaw_deg=dyaw,
        dtx=dtx,
        dty=dty,
        dtz=dtz,
    )

    current = evaluate_pose_candidate(
        big_labels=_G_BIG_LABELS,
        points_xyz=_G_POINTS_XYZ,
        point_labels=_G_POINT_LABELS,
        R=R_cand,
        t=t_cand,
        K=_G_K,
        image_size=_G_IMAGE_SIZE,
        class_ids=_G_CLASS_IDS,
        min_valid_pixels=_G_MIN_VALID_PIXELS,
    )

    return {
        "score": float(current["score"]),
        "num_valid_pixels": int(current["num_valid_pixels"]),
        "is_valid": bool(current["is_valid"]),
        "droll_deg": float(droll),
        "dpitch_deg": float(dpitch),
        "dyaw_deg": float(dyaw),
        "dtx": float(dtx),
        "dty": float(dty),
        "dtz": float(dtz),
    }

def generate_pose_grid(
    roll_values_deg,
    pitch_values_deg,
    yaw_values_deg,
    tx_values,
    ty_values,
    tz_values,
):
    for vals in product(
        roll_values_deg,
        pitch_values_deg,
        yaw_values_deg,
        tx_values,
        ty_values,
        tz_values,
    ):
        yield vals

def save_result_jsonl(jsonl_path, record):
    """
    Guarda resultados incrementales en JSONL.
    Solo escribe el proceso principal.
    """
'''    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
'''

def parallel_search_pose_3d(
    big_labels,
    points_xyz,
    point_labels,
    R0,
    t0,
    K,
    image_size,
    class_ids=VALID_CLASSES,
    roll_values_deg=(-2.0, 0.0, 2.0),
    pitch_values_deg=(-2.0, 0.0, 2.0),
    yaw_values_deg=(-5.0, 0.0, 5.0),
    tx_values=(-0.25, 0.0, 0.25),
    ty_values=(-0.25, 0.0, 0.25),
    tz_values=(-0.25, 0.0, 0.25),
    min_valid_pixels=50,
    max_workers=None,
    out_jsonl=None,
    verbose=True,
):
    """
    Búsqueda paralela de pose.
    Devuelve el mejor candidato ligero.
    """
    candidates = list(generate_pose_grid(
        roll_values_deg,
        pitch_values_deg,
        yaw_values_deg,
        tx_values,
        ty_values,
        tz_values,
    ))

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    best = None
    num_tested = 0
    num_valid = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(
            big_labels,
            points_xyz,
            point_labels,
            R0,
            t0,
            K,
            image_size,
            class_ids,
            min_valid_pixels,
        ),
    ) as ex:
        futures = {
            ex.submit(evaluate_pose_candidate_light, *cand): cand
            for cand in candidates
        }

        for fut in as_completed(futures):
            num_tested += 1
            rec = fut.result()

            if out_jsonl is not None:
                save_result_jsonl(out_jsonl, rec)

            if not rec["is_valid"]:
                continue

            num_valid += 1

            if best is None or rec["score"] > best["score"]:
                best = rec
                if verbose:
                    print(
                        "[parallel] nuevo mejor:"
                        f" score={best['score']:.6f}"
                        f" | droll={best['droll_deg']:.3f}"
                        f" | dpitch={best['dpitch_deg']:.3f}"
                        f" | dyaw={best['dyaw_deg']:.3f}"
                        f" | dtx={best['dtx']:.3f}"
                        f" | dty={best['dty']:.3f}"
                        f" | dtz={best['dtz']:.3f}"
                        f" | valid_pixels={best['num_valid_pixels']}"
                    )

    if best is None:
        raise RuntimeError("No se encontró ninguna pose válida.")

    best["num_tested"] = num_tested
    best["num_valid_candidates"] = num_valid
    return best

def rebuild_full_candidate(best_light, big_labels, points_xyz, point_labels, R0, t0, K, image_size):
    """
    Recalcula SOLO el mejor candidato para recuperar R, t, proj_bgr, proj_valid, etc.
    """
    R_best, t_best = compose_pose(
        R0=R0,
        t0=t0,
        droll_deg=best_light["droll_deg"],
        dpitch_deg=best_light["dpitch_deg"],
        dyaw_deg=best_light["dyaw_deg"],
        dtx=best_light["dtx"],
        dty=best_light["dty"],
        dtz=best_light["dtz"],
    )

    full = evaluate_pose_candidate(
        big_labels=big_labels,
        points_xyz=points_xyz,
        point_labels=point_labels,
        R=R_best,
        t=t_best,
        K=K,
        image_size=image_size,
        class_ids=VALID_CLASSES,
        min_valid_pixels=50,
    )

    full["droll_deg"] = best_light["droll_deg"]
    full["dpitch_deg"] = best_light["dpitch_deg"]
    full["dyaw_deg"] = best_light["dyaw_deg"]
    full["dtx"] = best_light["dtx"]
    full["dty"] = best_light["dty"]
    full["dtz"] = best_light["dtz"]
    return full

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

    color_to_class = {tuple(color): cls for cls, color in CLASS_COLOR_MAP.items()}

    for color, cls in color_to_class.items():
        mask = np.all(img_bgr == np.array(color, dtype=np.uint8), axis=2)
        labels[mask] = cls

    return labels

# =========================================================
# 4) Máscara válida
# =========================================================
def valid_region_from_labels(labels):
    return ((labels > 0).astype(np.uint8) * 255)

# =========================================================
# 5) Recorte bbox válida
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
# 6) Visualización de labels
# =========================================================
def colorize_labels(labels):
    out = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for c, color in CLASS_COLOR_MAP.items():
        out[labels == c] = color
    return out

# =========================================================
# 7) Overlay
# =========================================================
def build_overlay(big, patch, valid_mask, top_left, alpha=0.20):
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
# 8) Pegar patch
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

def euler_to_rotation_matrix(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """
    Construye una matriz de rotación 3D a partir de ángulos de Euler
    en grados.

    Convención usada:
      - roll  : rotación sobre eje X
      - pitch : rotación sobre eje Y
      - yaw   : rotación sobre eje Z

    La composición es:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Retorna
    -------
    R : np.ndarray (3, 3)
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx,  cx]
    ], dtype=np.float64)

    Ry = np.array([
        [ cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy]
    ], dtype=np.float64)

    Rz = np.array([
        [cz, -sz, 0.0],
        [sz,  cz, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    R = Rz @ Ry @ Rx
    return R

def compose_pose(
    R0,
    t0,
    droll_deg=0.0,
    dpitch_deg=0.0,
    dyaw_deg=0.0,
    dtx=0.0,
    dty=0.0,
    dtz=0.0
):
    """
    Compone una pose candidata a partir de la pose inicial (R0, t0)
    y pequeños incrementos angulares/traslacionales.

    Convención:
        R_cand = dR @ R0
        t_cand = t0 + dt

    Parámetros
    ----------
    R0 : np.ndarray (3, 3)
        Rotación inicial.
    t0 : np.ndarray (3,)
        Traslación inicial.
    droll_deg, dpitch_deg, dyaw_deg : float
        Incrementos angulares en grados.
    dtx, dty, dtz : float
        Incrementos de traslación en las unidades de t0.

    Retorna
    -------
    R_cand : np.ndarray (3, 3)
    t_cand : np.ndarray (3,)
    """
    dR = euler_to_rotation_matrix(
        roll_deg=droll_deg,
        pitch_deg=dpitch_deg,
        yaw_deg=dyaw_deg
    )

    dt = np.array([dtx, dty, dtz], dtype=np.float64)

    R_cand = dR @ np.asarray(R0, dtype=np.float64)
    t_cand = np.asarray(t0, dtype=np.float64).reshape(3) + dt

    return R_cand, t_cand

def dilate_mask(mask, radius):
    if radius <= 0:
        return mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

def compute_per_class_iou_stats(
    big_labels,
    proj_labels,
    class_ids=VALID_CLASSES,
    class_weights=None,
    valid_mask=None,
    dilate_radius=1,
):
    """
    Calcula IoU por clase y devuelve un dict detallado.
    Si dilate_radius > 0, dilata ambas máscaras antes de comparar.
    """
    if class_weights is None:
        class_weights = {c: 1.0 for c in class_ids}

    if valid_mask is None:
        valid_mask = (proj_labels > 0).astype(np.uint8)
    else:
        valid_mask = (valid_mask > 0).astype(np.uint8)

    if dilate_radius > 0:
        valid_mask = dilate_mask(valid_mask, dilate_radius)

    stats = {}
    for c in class_ids:
        big_c = ((big_labels == c).astype(np.uint8))
        proj_c = ((proj_labels == c).astype(np.uint8))

        if dilate_radius > 0:
            big_c = dilate_mask(big_c, dilate_radius)
            proj_c = dilate_mask(proj_c, dilate_radius)

        big_c = big_c.astype(bool) & valid_mask.astype(bool)
        proj_c = proj_c.astype(bool) & valid_mask.astype(bool)

        inter = int(np.logical_and(big_c, proj_c).sum())
        union = int(np.logical_or(big_c, proj_c).sum())
        cam_pixels = int(big_c.sum())
        proj_pixels = int(proj_c.sum())
        iou = float(inter / union) if union > 0 else 0.0

        stats[int(c)] = {
            "weight": float(class_weights.get(c, 0.0)),
            "cam_pixels": cam_pixels,
            "proj_pixels": proj_pixels,
            "intersection": inter,
            "union": union,
            "iou": iou,
        }
    return stats

def summarize_weighted_score_from_stats(stats):
    total_score = 0.0
    total_weight = 0.0
    for s in stats.values():
        w = float(s["weight"])
        if w <= 0.0 or int(s["union"]) == 0:
            continue
        total_score += w * float(s["iou"])
        total_weight += w
    if total_weight <= 0.0:
        return 0.0
    return total_score / total_weight

def save_candidate_visuals(
    stage_name,
    stage_result,
    img_original,
    big_labels,
    out_dir,
    points_xyz=None,
    point_labels=None,
    K=None,
    image_size=None,
):
    """
    Guarda patch_labels, overlay y composite para una etapa dada.

    Si se pasan points_xyz / point_labels / K / image_size, vuelve a renderizar
    SOLO para visualización con point_radius=0, sin afectar al score ni a la
    pose óptima.
    """
    if (
        points_xyz is not None and
        point_labels is not None and
        K is not None and
        image_size is not None
    ):
        proj_labels_vis, proj_bgr_vis, proj_valid_vis = render_lidar_patch_from_projection(
            points_xyz=points_xyz,
            point_labels=point_labels,
            R=stage_result["R"],
            t=stage_result["t"],
            K=K,
            image_size=image_size,
            point_radius=2
        )
    else:
        proj_bgr_vis = stage_result["proj_bgr"]
        proj_valid_vis = stage_result["proj_valid"]

    proj_bgr_up, proj_valid_up, top_left_up = upscale_patch_and_mask_to_original(
        proj_bgr_vis,
        proj_valid_vis,
        (0, 0),
        original_shape=img_original.shape,
        working_shape=big_labels.shape
    )

    overlay = build_overlay(
        img_original,
        proj_bgr_up,
        proj_valid_up,
        top_left_up,
        alpha=0.25
    )

    composite = paste_with_valid_mask_clipped(
        img_original,
        proj_bgr_up,
        proj_valid_up,
        top_left_up
    )

    if stage_name == "initial":
        cv2.imwrite(
            os.path.join(out_dir, "overlay_initial_on_rgb.png"),
            overlay
        )

    elif stage_name == "refined":
        cv2.imwrite(
            os.path.join(out_dir, "overlay_refined_on_rgb.png"),
            overlay
        )
        cv2.imwrite(
            os.path.join(out_dir, "composite_refined_on_rgb.png"),
            composite
        )

def save_stage_diagnostics(stage_name, stage_result, out_dir):
    '''data = {
        "score": float(stage_result["score"]),
        "score_strict": float(stage_result.get("score_strict", stage_result["score"])),
        "score_soft": float(stage_result.get("score_soft", stage_result["score"])),
        "num_valid_pixels": int(stage_result["num_valid_pixels"]),
        "class_weights": {
            int(k): float(v) for k, v in stage_result.get("class_weights", {}).items()
        },
        "per_class_iou_strict": stage_result.get("per_class_iou_strict", {}),
        "per_class_iou_soft": stage_result.get("per_class_iou_soft", {}),
    }
    with open(os.path.join(out_dir, f"diagnostics_{stage_name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)'''
    return

def evaluate_pose_candidate(
    big_labels,
    points_xyz,
    point_labels,
    R,
    t,
    K,
    image_size,
    class_ids=VALID_CLASSES,
    min_valid_pixels=50
):
    """
    Evalúa una pose candidata con una métrica híbrida:
      - score_strict: IoU multiclase estricta
      - score_soft  : IoU multiclase tras pequeña dilatación
      - score final : combinación de ambas

    Así la métrica correlaciona mejor con la calidad visual y no depende
    tanto de coincidencias de 1 píxel.
    """
    proj_labels, proj_bgr, proj_valid = render_lidar_patch_from_projection(
        points_xyz=points_xyz,
        point_labels=point_labels,
        R=R,
        t=t,
        K=K,
        image_size=image_size
    )

    num_valid = int(np.count_nonzero(proj_valid))

    if num_valid < min_valid_pixels:
        return {
            "score": 0.0,
            "score_strict": 0.0,
            "score_soft": 0.0,
            "num_valid_pixels": num_valid,
            "proj_labels": proj_labels,
            "proj_bgr": proj_bgr,
            "proj_valid": proj_valid,
            "class_weights": {c: 0.0 for c in class_ids},
            "per_class_iou_strict": {},
            "per_class_iou_soft": {},
            "R": R.copy(),
            "t": t.copy(),
            "is_valid": False,
        }

    class_weights = compute_class_weights_from_patch(
        proj_labels,
        class_ids
    )

    strict_stats = compute_per_class_iou_stats(
        big_labels=big_labels,
        proj_labels=proj_labels,
        class_ids=class_ids,
        class_weights=class_weights,
        valid_mask=proj_valid,
        dilate_radius=0,
    )
    score_strict = summarize_weighted_score_from_stats(strict_stats)

    soft_stats = compute_per_class_iou_stats(
        big_labels=big_labels,
        proj_labels=proj_labels,
        class_ids=class_ids,
        class_weights=class_weights,
        valid_mask=proj_valid,
        dilate_radius=3,
    )
    score_soft = summarize_weighted_score_from_stats(soft_stats)

    # Combinación: prioriza semántica estricta pero añade tolerancia espacial.
    score = 0.65 * score_strict + 0.35 * score_soft

    return {
        "score": float(score),
        "score_strict": float(score_strict),
        "score_soft": float(score_soft),
        "num_valid_pixels": num_valid,
        "proj_labels": proj_labels,
        "proj_bgr": proj_bgr,
        "proj_valid": proj_valid,
        "class_weights": class_weights,
        "per_class_iou_strict": strict_stats,
        "per_class_iou_soft": soft_stats,
        "R": R.copy(),
        "t": t.copy(),
        "is_valid": True,
    }

def weighted_multiclass_iou_full(

    big_labels,
    proj_labels,
    class_ids=VALID_CLASSES,
    class_weights=None,
    valid_mask=None
):
    """
    IoU multiclase ponderada entre:
      - big_labels: segmentación cámara (H, W)
      - proj_labels: proyección LiDAR ya renderizada en imagen (H, W)

    La comparación se hace solo en la región válida de la proyección
    (por defecto, donde proj_labels > 0), para no penalizar zonas donde
    no hay puntos LiDAR.
    """
    if big_labels.shape != proj_labels.shape:
        raise ValueError(
            f"Shapes incompatibles: big_labels={big_labels.shape}, "
            f"proj_labels={proj_labels.shape}"
        )

    if class_weights is None:
        class_weights = {c: 1.0 for c in class_ids}

    if valid_mask is None:
        valid_mask = (proj_labels > 0)
    else:
        valid_mask = (valid_mask > 0)

    if np.count_nonzero(valid_mask) == 0:
        return 0.0

    total_score = 0.0
    total_weight = 0.0

    for c in class_ids:
        w_c = float(class_weights.get(c, 0.0))
        if w_c <= 0.0:
            continue

        big_c = (big_labels == c) & valid_mask
        proj_c = (proj_labels == c) & valid_mask

        union = np.logical_or(big_c, proj_c).sum()
        if union == 0:
            continue

        inter = np.logical_and(big_c, proj_c).sum()
        iou_c = inter / union

        total_score += w_c * iou_c
        total_weight += w_c

    if total_weight == 0.0:
        return 0.0

    return total_score / total_weight

def coarse_search_pose_3d(
    big_labels,
    points_xyz,
    point_labels,
    R0,
    t0,
    K,
    image_size,
    class_ids=VALID_CLASSES,
    roll_values_deg=(-2.0,0.0,2.0),
    pitch_values_deg=(-2.0, 0.0, 2.0),
    yaw_values_deg=(-5.0,0.0,5.0),
    tx_values=(-0.25, 0.0, 0.25),
    ty_values=(-0.25, 0.0, 0.25),
    tz_values = (-0.25, 0.0, 0.25),
    min_valid_pixels=50,
    verbose=False
):
    """
    Búsqueda gruesa 3D sobre incrementos de pose alrededor de (R0, t0).

    Retorna el mejor candidato encontrado, incluyendo score, pose,
    render y deltas usados.
    """
    best = None
    num_tested = 0
    num_valid = 0

    for droll in roll_values_deg:
        for dpitch in pitch_values_deg:
            for dyaw in yaw_values_deg:
                for dtx in tx_values:
                    for dty in ty_values:
                        for dtz in tz_values:
                            num_tested += 1

                            R_cand, t_cand = compose_pose(
                                R0=R0,
                                t0=t0,
                                droll_deg=droll,
                                dpitch_deg=dpitch,
                                dyaw_deg=dyaw,
                                dtx=dtx,
                                dty=dty,
                                dtz=dtz
                            )

                            current = evaluate_pose_candidate(
                                big_labels=big_labels,
                                points_xyz=points_xyz,
                                point_labels=point_labels,
                                R=R_cand,
                                t=t_cand,
                                K=K,
                                image_size=image_size,
                                class_ids=class_ids,
                                min_valid_pixels=min_valid_pixels
                            )

                            current["droll_deg"] = float(droll)
                            current["dpitch_deg"] = float(dpitch)
                            current["dyaw_deg"] = float(dyaw)
                            current["dtx"] = float(dtx)
                            current["dty"] = float(dty)
                            current["dtz"] = float(dtz)

                            if not current["is_valid"]:
                                continue

                            num_valid += 1

                            if (best is None) or (current["score"] > best["score"]):
                                best = current

                                if verbose:
                                    print(
                                        "[coarse 3D] nuevo mejor:"
                                        f" score={best['score']:.6f}"
                                        f" | droll={droll:.3f}"
                                        f" | dpitch={dpitch:.3f}"
                                        f" | dyaw={dyaw:.3f}"
                                        f" | dtx={dtx:.3f}"
                                        f" | dty={dty:.3f}"
                                        f" | dtz={dtz:.3f}"
                                        f" | valid_pixels={best['num_valid_pixels']}"
                                    )

    if best is None:
        raise RuntimeError(
            "No se encontró ninguna pose válida en coarse_search_pose_3d."
        )

    best["num_tested"] = num_tested
    best["num_valid_candidates"] = num_valid

    if verbose:
        print("\n=== RESULTADO BÚSQUEDA GRUESA 3D ===")
        print("candidatos probados :", num_tested)
        print("candidatos válidos  :", num_valid)
        print("best score          :", best["score"])
        print("best droll_deg      :", best["droll_deg"])
        print("best dpitch_deg     :", best["dpitch_deg"])
        print("best dyaw_deg       :", best["dyaw_deg"])
        print("best dtx            :", best["dtx"])
        print("best dty            :", best["dty"])
        print("best dtz            :", best["dtz"])

    return best

def refine_pose_3d(
    big_labels,
    points_xyz,
    point_labels,
    R0,
    t0,
    K,
    image_size,
    coarse_best,
    class_ids=VALID_CLASSES,
    roll_step_deg=0.5,
    pitch_step_deg=0.5,
    yaw_step_deg=1.0,
    tx_step=0.05,
    ty_step=0.05,
    tz_step=0.05,
    min_valid_pixels=50,
    verbose=False
):
    """
    Refinamiento fino 3D alrededor del mejor candidato coarse.

    coarse_best debe venir de coarse_search_pose_3d(...) y contener:
      - droll_deg, dpitch_deg, dyaw_deg
      - dtx, dty, dtz

    La búsqueda fina se hace en una vecindad 3x3x3x3x3x3 alrededor
    de esos deltas.
    """
    center_droll = float(coarse_best["droll_deg"])
    center_dpitch = float(coarse_best["dpitch_deg"])
    center_dyaw = float(coarse_best["dyaw_deg"])
    center_dtx = float(coarse_best["dtx"])
    center_dty = float(coarse_best["dty"])
    center_dtz = float(coarse_best["dtz"])

    roll_values_deg = [
        center_droll - roll_step_deg,
        center_droll,
        center_droll + roll_step_deg,
    ]
    pitch_values_deg = [
        center_dpitch - pitch_step_deg,
        center_dpitch,
        center_dpitch + pitch_step_deg,
    ]
    yaw_values_deg = [
        center_dyaw - yaw_step_deg,
        center_dyaw,
        center_dyaw + yaw_step_deg,
    ]

    tx_values = [
        center_dtx - tx_step,
        center_dtx,
        center_dtx + tx_step,
    ]
    ty_values = [
        center_dty - ty_step,
        center_dty,
        center_dty + ty_step,
    ]
    tz_values = [
        center_dtz - tz_step,
        center_dtz,
        center_dtz + tz_step,
    ]

    best = None
    num_tested = 0
    num_valid = 0

    for droll in roll_values_deg:
        for dpitch in pitch_values_deg:
            for dyaw in yaw_values_deg:
                for dtx in tx_values:
                    for dty in ty_values:
                        for dtz in tz_values:
                            num_tested += 1

                            R_cand, t_cand = compose_pose(
                                R0=R0,
                                t0=t0,
                                droll_deg=droll,
                                dpitch_deg=dpitch,
                                dyaw_deg=dyaw,
                                dtx=dtx,
                                dty=dty,
                                dtz=dtz
                            )

                            current = evaluate_pose_candidate(
                                big_labels=big_labels,
                                points_xyz=points_xyz,
                                point_labels=point_labels,
                                R=R_cand,
                                t=t_cand,
                                K=K,
                                image_size=image_size,
                                class_ids=class_ids,
                                min_valid_pixels=min_valid_pixels
                            )

                            current["droll_deg"] = float(droll)
                            current["dpitch_deg"] = float(dpitch)
                            current["dyaw_deg"] = float(dyaw)
                            current["dtx"] = float(dtx)
                            current["dty"] = float(dty)
                            current["dtz"] = float(dtz)

                            if not current["is_valid"]:
                                continue

                            num_valid += 1

                            if (best is None) or (current["score"] > best["score"]):
                                best = current

                                if verbose:
                                    print(
                                        "[refine 3D] nuevo mejor:"
                                        f" score={best['score']:.6f}"
                                        f" | droll={droll:.3f}"
                                        f" | dpitch={dpitch:.3f}"
                                        f" | dyaw={dyaw:.3f}"
                                        f" | dtx={dtx:.3f}"
                                        f" | dty={dty:.3f}"
                                        f" | dtz={dtz:.3f}"
                                        f" | valid_pixels={best['num_valid_pixels']}"
                                    )

    if best is None:
        raise RuntimeError(
            "No se encontró ninguna pose válida en refine_pose_3d."
        )

    best["num_tested"] = num_tested
    best["num_valid_candidates"] = num_valid

    if verbose:
        print("\n=== RESULTADO REFINADO 3D ===")
        print("candidatos probados :", num_tested)
        print("candidatos válidos  :", num_valid)
        print("best score          :", best["score"])
        print("best droll_deg      :", best["droll_deg"])
        print("best dpitch_deg     :", best["dpitch_deg"])
        print("best dyaw_deg       :", best["dyaw_deg"])
        print("best dtx            :", best["dtx"])
        print("best dty            :", best["dty"])
        print("best dtz            :", best["dtz"])

    return best

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
    image_size,
    point_radius=1
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

        x0 = max(0, x - point_radius)
        x1 = min(W, x + point_radius + 1)
        y0 = max(0, y - point_radius)
        y1 = min(H, y + point_radius + 1)

        # z-buffer por vecindad local para evitar una proyección demasiado rala
        region_depth = depth_buffer[y0:y1, x0:x1]
        region_labels = patch_labels_full[y0:y1, x0:x1]

        mask_update = z < region_depth
        region_depth[mask_update] = z
        region_labels[mask_update] = np.uint8(lbl)

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
# 13) Main
# =========================================================
def main(
    calib_path,
    image_path,
    rgb_seg_path,
    points_path,
    labels_path,
    out_dir="resultados_lidar_GOOSE_3D_parallel",
    max_workers=None
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

    # image_size del json viene como [W, H]
    W_calib, H_calib = int(image_size[0]), int(image_size[1])
    seg_h, seg_w = rgb_seg.shape[:2]

    K_work = K.copy().astype(np.float64)

    # -----------------------------------------------------
    # 2) Ajuste de intrínseca al tamaño de la segmentación
    # -----------------------------------------------------
    if (seg_h, seg_w) != (H_calib, W_calib):
        print("[AVISO] rgb_seg.shape y image_size de calibración no coinciden.")
        print("rgb_seg.shape         :", rgb_seg.shape)
        print("image_size json [W,H] :", image_size)

        scale_x = seg_w / W_calib
        scale_y = seg_h / H_calib

        print(f"[INFO] Escalando K con scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")

        K_work[0, 0] *= scale_x   # fx
        K_work[0, 2] *= scale_x   # cx
        K_work[1, 1] *= scale_y   # fy
        K_work[1, 2] *= scale_y   # cy

    image_size_hw = (seg_h, seg_w)

    print("=== DATOS CARGADOS ===")
    print("RGB original shape      :", img_original.shape)
    print("SEG shape               :", rgb_seg.shape)
    print("Puntos LiDAR            :", points_xyz.shape)
    print("Labels LiDAR            :", lidar_labels.shape)
    print("Calib image_size [W,H]  :", image_size)
    print("Working image_size [H,W]:", image_size_hw)

    # -----------------------------------------------------
    # 3) Big = segmentación 2D de cámara
    # -----------------------------------------------------
    big_labels = rgb_seg

    # -----------------------------------------------------
    # 4) Evaluación inicial
    # -----------------------------------------------------
    initial = evaluate_pose_candidate(
        big_labels=big_labels,
        points_xyz=points_xyz,
        point_labels=lidar_labels,
        R=R0,
        t=t0,
        K=K_work,
        image_size=image_size_hw,
        class_ids=VALID_CLASSES,
        min_valid_pixels=50
    )

    print("\n=== EVALUACIÓN INICIAL ===")
    print("score inicial       :", initial["score"])
    print("score strict inicial:", initial["score_strict"])
    print("score soft inicial  :", initial["score_soft"])
    print("valid pixels inicial:", initial["num_valid_pixels"])

    initial_proj_bgr_up, initial_proj_valid_up, initial_top_left_up = upscale_patch_and_mask_to_original(
        initial["proj_bgr"],
        initial["proj_valid"],
        (0, 0),
        original_shape=img_original.shape,
        working_shape=big_labels.shape
    )

    overlay_initial = build_overlay(
        img_original,
        initial_proj_bgr_up,
        initial_proj_valid_up,
        initial_top_left_up,
        alpha=0.20
    )
    save_candidate_visuals(
    "initial",
    initial,
    img_original,
    big_labels,
    out_dir,
    points_xyz=points_xyz,
    point_labels=lidar_labels,
    K=K_work,
    image_size=image_size_hw
)
    save_stage_diagnostics("initial", initial, out_dir)

    # -----------------------------------------------------
    # 5) Búsqueda gruesa paralela
    # -----------------------------------------------------
    coarse_light = parallel_search_pose_3d(
        big_labels=big_labels,
        points_xyz=points_xyz,
        point_labels=lidar_labels,
        R0=R0,
        t0=t0,
        K=K_work,
        image_size=image_size_hw,
        class_ids=VALID_CLASSES,
        roll_values_deg=(-2.0, 0.0, 2.0),
        pitch_values_deg=(-2.0, 0.0, 2.0),
        yaw_values_deg=(-5.0, 0.0, 5.0),
        tx_values=(-0.2, 0.0, 0.2),
        ty_values=(-0.2, 0.0, 0.2),
        tz_values=(-0.2, 0.0, 0.2),
        min_valid_pixels=50,
        max_workers=max_workers,
        out_jsonl=os.path.join(out_dir, "coarse_results.jsonl"),
        verbose=False
    )

    coarse = rebuild_full_candidate(
        coarse_light,
        big_labels,
        points_xyz,
        lidar_labels,
        R0,
        t0,
        K_work,
        image_size_hw
    )

    print("\n=== RESULTADO BÚSQUEDA GRUESA 3D ===")
    print("score coarse    :", coarse["score"])
    print("score strict    :", coarse["score_strict"])
    print("score soft      :", coarse["score_soft"])
    print("droll_deg       :", coarse["droll_deg"])
    print("dpitch_deg      :", coarse["dpitch_deg"])
    print("dyaw_deg        :", coarse["dyaw_deg"])
    print("dtx             :", coarse["dtx"])
    print("dty             :", coarse["dty"])
    print("dtz             :", coarse["dtz"])
    print("num_tested      :", coarse_light["num_tested"])
    print("num_valid       :", coarse_light["num_valid_candidates"])

    # -----------------------------------------------------
    # 6) Refinado paralelo
    # -----------------------------------------------------
    refine_roll = [
        coarse_light["droll_deg"] - 0.5,
        coarse_light["droll_deg"],
        coarse_light["droll_deg"] + 0.5,
    ]
    refine_pitch = [
        coarse_light["dpitch_deg"] - 0.5,
        coarse_light["dpitch_deg"],
        coarse_light["dpitch_deg"] + 0.5,
    ]
    refine_yaw = [
        coarse_light["dyaw_deg"] - 1.0,
        coarse_light["dyaw_deg"],
        coarse_light["dyaw_deg"] + 1.0,
    ]
    refine_tx = [
        coarse_light["dtx"] - 0.05,
        coarse_light["dtx"],
        coarse_light["dtx"] + 0.05,
    ]
    refine_ty = [
        coarse_light["dty"] - 0.05,
        coarse_light["dty"],
        coarse_light["dty"] + 0.05,
    ]
    refine_tz = [
        coarse_light["dtz"] - 0.05,
        coarse_light["dtz"],
        coarse_light["dtz"] + 0.05,
    ]

    refined_light = parallel_search_pose_3d(
        big_labels=big_labels,
        points_xyz=points_xyz,
        point_labels=lidar_labels,
        R0=R0,
        t0=t0,
        K=K_work,
        image_size=image_size_hw,
        class_ids=VALID_CLASSES,
        roll_values_deg=refine_roll,
        pitch_values_deg=refine_pitch,
        yaw_values_deg=refine_yaw,
        tx_values=refine_tx,
        ty_values=refine_ty,
        tz_values=refine_tz,
        min_valid_pixels=50,
        max_workers=max_workers,
        out_jsonl=os.path.join(out_dir, "refine_results.jsonl"),
        verbose=False
    )

    refined = rebuild_full_candidate(
        refined_light,
        big_labels,
        points_xyz,
        lidar_labels,
        R0,
        t0,
        K_work,
        image_size_hw
    )

    print("\n=== RESULTADO REFINADO 3D ===")
    print("best score      :", refined["score"])
    print("best score strict:", refined["score_strict"])
    print("best score soft  :", refined["score_soft"])
    print("best droll_deg  :", refined["droll_deg"])
    print("best dpitch_deg :", refined["dpitch_deg"])
    print("best dyaw_deg   :", refined["dyaw_deg"])
    print("best dtx        :", refined["dtx"])
    print("best dty        :", refined["dty"])
    print("best dtz        :", refined["dtz"])
    print("num_tested      :", refined_light["num_tested"])
    print("num_valid       :", refined_light["num_valid_candidates"])

    # -----------------------------------------------------
    # 7) Guardar visualizaciones y diagnóstico por etapa
    # -----------------------------------------------------
    save_candidate_visuals(
        "coarse",
        coarse,
        img_original,
        big_labels,
        out_dir,
        points_xyz=points_xyz,
        point_labels=lidar_labels,
        K=K_work,
        image_size=image_size_hw
    )  
    save_stage_diagnostics("coarse", coarse, out_dir)

    save_candidate_visuals(
    "refined",
    refined,
    img_original,
    big_labels,
    out_dir,
    points_xyz=points_xyz,
    point_labels=lidar_labels,
    K=K_work,
    image_size=image_size_hw
    )
    save_stage_diagnostics("refined", refined, out_dir)

    # -----------------------------------------------------
    # 8) Guardar resultado
    # -----------------------------------------------------
    result = {
        "score_initial": float(initial["score"]),
        "score_coarse": float(coarse["score"]),
        "score_refined": float(refined["score"]),
        "score_initial_strict": float(initial["score_strict"]),
        "score_initial_soft": float(initial["score_soft"]),
        "score_coarse_strict": float(coarse["score_strict"]),
        "score_coarse_soft": float(coarse["score_soft"]),
        "score_refined_strict": float(refined["score_strict"]),
        "score_refined_soft": float(refined["score_soft"]),
        "delta_pose": {
            "droll_deg": float(refined["droll_deg"]),
            "dpitch_deg": float(refined["dpitch_deg"]),
            "dyaw_deg": float(refined["dyaw_deg"]),
            "dtx": float(refined["dtx"]),
            "dty": float(refined["dty"]),
            "dtz": float(refined["dtz"]),
        },
        "R0": np.asarray(R0, dtype=float).tolist(),
        "t0": np.asarray(t0, dtype=float).reshape(3).tolist(),
        "R_refined": np.asarray(refined["R"], dtype=float).tolist(),
        "t_refined": np.asarray(refined["t"], dtype=float).reshape(3).tolist(),
        "num_valid_pixels_initial": int(initial["num_valid_pixels"]),
        "num_valid_pixels_coarse": int(coarse["num_valid_pixels"]),
        "num_valid_pixels_refined": int(refined["num_valid_pixels"]),
        "coarse_search_stats": {
            "num_tested": int(coarse_light["num_tested"]),
            "num_valid_candidates": int(coarse_light["num_valid_candidates"]),
        },
        "refine_search_stats": {
            "num_tested": int(refined_light["num_tested"]),
            "num_valid_candidates": int(refined_light["num_valid_candidates"]),
        },
        "class_weights": {
            int(k): float(v) for k, v in refined["class_weights"].items()
        },
        "per_class_iou_strict_refined": refined["per_class_iou_strict"],
        "per_class_iou_soft_refined": refined["per_class_iou_soft"]
    }

    with open(os.path.join(out_dir, "alignment_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return {
        "calib_path": calib_path,
        "image_path": image_path,
        "rgb_seg_path": rgb_seg_path,
        "points_path": points_path,
        "labels_path": labels_path,
        "initial_score": initial["score"],
        "coarse_score": coarse["score"],
        "refined_score": refined["score"],
        "delta_pose": {
            "droll_deg": refined["droll_deg"],
            "dpitch_deg": refined["dpitch_deg"],
            "dyaw_deg": refined["dyaw_deg"],
            "dtx": refined["dtx"],
            "dty": refined["dty"],
            "dtz": refined["dtz"],
        },
        "R_refined": refined["R"],
        "t_refined": refined["t"],
        "out_dir": out_dir,
    }
if __name__ == "__main__":
    main(
        calib_path="test_GOOSE/2022-08-30_siegertsbrunn_feldwege-0610_1661861076763438335-calib.json",
        image_path="test_GOOSE/2022-08-30_siegertsbrunn_feldwege-0610_1661861076763438335-image.png",
        rgb_seg_path="test_GOOSE/2022-08-30_siegertsbrunn_feldwege-0610_1661861076763438335-pred.png",
        points_path="test_GOOSE/2022-08-30_siegertsbrunn_feldwege-0610_1661861076763438335-points.bin",
        labels_path="test_GOOSE/2022-08-30_siegertsbrunn_feldwege-0610_1661861076763438335-pred.label",
        out_dir="resultados_lidar_GOOSE_3D_parallel_5",
        max_workers=None  # o por ejemplo 8
    )

