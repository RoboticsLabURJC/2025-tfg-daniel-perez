import json
import cv2
import numpy as np



# CARGA DE DATOS

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

# GEOMETRÍA

def rodrigues_from_euler(rx, ry, rz):
    """
    Euler -> matriz de rotación.
    Orden: Rz @ Ry @ Rx
    Ángulos en radianes.
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx]
    ], dtype=np.float64)

    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy]
    ], dtype=np.float64)

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [0,   0,  1]
    ], dtype=np.float64)

    return Rz @ Ry @ Rx

def random_perturbation(rot_range_deg=2.0, trans_range=0.05):
    """
    Perturbación aleatoria pequeña alrededor de la calibración inicial.
    """
    angles_deg = np.random.uniform(-rot_range_deg, rot_range_deg, size=3)
    angles_rad = np.deg2rad(angles_deg)

    dR = rodrigues_from_euler(*angles_rad)
    dt = np.random.uniform(-trans_range, trans_range, size=3)

    return dR, dt, angles_deg

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

# LOSS

def semantic_alignment_loss_chamfer(
    uv,
    z_cam,
    lidar_labels,
    rgb_seg,
    class_ids=None,
    ignore_labels=None,
    squared=True,
    normalize=True,
    class_average=True,
    min_class_pixels=20,
    min_class_points=20,
    class_count_power=0.0
):
    """
    Chamfer unidireccional por clase:
    para cada punto LiDAR proyectado de una clase c,
    se toma la distancia al píxel 2D más cercano de esa misma clase.

    CORRECCIÓN:
    - evita que una clase con muchísimos puntos domine la loss
    - promedia por clase en vez de por punto global
    - ignora clases demasiado pequeñas/inestables
    - pondera suavemente por sqrt(n_puntos_clase)

    Parámetros nuevos
    -----------------
    class_average : bool
        Si True, combina por media ponderada de clases.
        Si False, usa media global por puntos.
    min_class_pixels : int
        Mínimo de píxeles 2D de una clase para usarla.
    min_class_points : int
        Mínimo de puntos LiDAR válidos de una clase para usarla.
    class_count_power : float
        Peso por clase = count ** class_count_power.
        0.0 -> todas iguales
        0.5 -> sqrt(count), recomendado
        1.0 -> proporcional a count
    """
    if ignore_labels is None:
        ignore_labels = set()

    H, W = rgb_seg.shape

    if class_ids is None:
        class_ids = sorted(int(c) for c in np.unique(lidar_labels) if int(c) not in ignore_labels)
    else:
        class_ids = [int(c) for c in class_ids if int(c) not in ignore_labels]

    total_loss = 0.0
    total_count = 0
    per_class_info = {}

    # Distance transform por clase
    dt_maps = {}
    class_pixel_counts = {}
    for cls in class_ids:
        mask = (rgb_seg == cls).astype(np.uint8)
        pix_count = int(mask.sum())
        class_pixel_counts[cls] = pix_count

        # ignorar clases demasiado pequeñas en 2D
        if pix_count < min_class_pixels:
            continue

        inv = 1 - mask
        dt = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
        dt_maps[cls] = dt

    # Acumular por clase
    for cls in class_ids:
        if cls not in dt_maps:
            continue

        dt = dt_maps[cls]
        cls_loss = 0.0
        cls_count = 0

        cls_mask = (lidar_labels == cls)

        for (u, v), z, keep in zip(uv, z_cam, cls_mask):
            if not keep:
                continue
            if z <= 0:
                continue
            if not np.isfinite(u) or not np.isfinite(v):
                continue

            x = int(round(u))
            y = int(round(v))

            if x < 0 or x >= W or y < 0 or y >= H:
                continue

            d = float(dt[y, x])
            if squared:
                d = d * d

            cls_loss += d
            cls_count += 1

        # ignorar clases demasiado pequeñas en 3D
        if cls_count < min_class_points:
            continue

        cls_mean = cls_loss / cls_count

        per_class_info[cls] = {
            "loss_sum": cls_loss,
            "count": cls_count,
            "loss_mean": cls_mean,
            "pixel_count_2d": class_pixel_counts.get(cls, 0)
        }

        total_loss += cls_loss
        total_count += cls_count

    if len(per_class_info) == 0:
        return np.inf, 0, {}

    # Opción 1: media por clases (recomendada)
    if class_average:
        weighted_sum = 0.0
        weight_sum = 0.0

        for cls, info in per_class_info.items():
            w = float(info["count"]) ** float(class_count_power)
            weighted_sum += w * float(info["loss_mean"])
            weight_sum += w

        if weight_sum <= 0:
            return np.inf, 0, {}

        total_loss = weighted_sum / weight_sum

    # Opción 2: media global por puntos
    else:
        if total_count == 0:
            return np.inf, 0, {}
        if normalize:
            total_loss = total_loss / total_count

    return total_loss, total_count, per_class_info

# BÚSQUEDA TIPO "LOSS-GUIDED INIT"

def compute_dynamic_class_weights(per_class_info, eps=1e-6, power=1.0):
    """
    Calcula pesos dinámicos por clase para balancear la loss.

    Idea:
    - clases con loss_mean alta pesan menos
    - clases con loss_mean baja pesan más
    - normalizamos para que la suma de pesos sea igual al nº de clases

    power=1.0  -> inverso directo
    power=0.5  -> más suave
    """
    valid_classes = [c for c, info in per_class_info.items() if info["count"] > 0]

    if len(valid_classes) == 0:
        return {}

    raw_weights = {}
    for c in valid_classes:
        loss_mean = float(per_class_info[c]["loss_mean"])
        raw_weights[c] = 1.0 / ((loss_mean + eps) ** power)

    s = sum(raw_weights.values())
    n = len(valid_classes)

    weights = {c: (raw_weights[c] / s) * n for c in valid_classes}
    return weights

'''def combine_relative_class_losses(per_class_info, base_class_info, eps=1e-6):
    total = 0.0
    used = 0

    for cls, info in per_class_info.items():
        if cls not in base_class_info:
            continue
        if info["count"] <= 0 or base_class_info[cls]["count"] <= 0:
            continue

        curr = float(info["loss_mean"])
        base = float(base_class_info[cls]["loss_mean"])

        total += curr / (base + eps)
        used += 1

    if used == 0:
        return np.inf

    return total / used'''

def search_best_calibration(
    points_xyz,
    lidar_labels,
    rgb_seg,
    R0,
    t0,
    K,
    N=400,
    rot_range_deg=2.0,
    trans_range=0.05,
    ignore_labels=None,
    class_ids=None,
    seed=0
):
    np.random.seed(seed)

    best = {
        "loss": np.inf,
        "raw_loss": np.inf,
        "R": None,
        "t": None,
        "angles_deg": None,
        "valid_points": 0,
        "per_class_info": {},
        "class_weights": {}
    }

    # evaluar calibración base
    uv0, z0 = project_points(points_xyz, R0, t0, K)
    loss0_raw, valid0, base_info = semantic_alignment_loss_chamfer(
        uv0,
        z0,
        lidar_labels,
        rgb_seg,
        class_ids=class_ids,
        ignore_labels=ignore_labels,
        squared=True,
        normalize=True,
        class_average=True,
        min_class_pixels=30,
        min_class_points=30,
        class_count_power=0.0
    )

    best["loss"] = loss0_raw
    best["raw_loss"] = loss0_raw
    best["R"] = R0.copy()
    best["t"] = t0.copy()
    best["angles_deg"] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    best["valid_points"] = valid0
    best["per_class_info"] = base_info
    best["class_weights"] = {}

    print(f"[BASE] loss={loss0_raw:.6f} valid_points={valid0}")
    print(f"[BASE] per_class_info={base_info}")

    for i in range(N):
        dR, dt, angles_deg = random_perturbation(
            rot_range_deg=rot_range_deg,
            trans_range=trans_range
        )

        # perturbación alrededor de la calibración inicial
        R = dR @ R0
        t = t0 + dt

        uv, z_cam = project_points(points_xyz, R, t, K)

        loss_raw, valid_pts, info_cls = semantic_alignment_loss_chamfer(
            uv,
            z_cam,
            lidar_labels,
            rgb_seg,
            class_ids=class_ids,
            ignore_labels=ignore_labels,
            squared=True,
            normalize=True,
            class_average=True,
            min_class_pixels=30,
            min_class_points=30,
            class_count_power=0.0
        )

        print(
            f"[{i+1:03d}/{N}] "
            f"loss={loss_raw:.6f} "
            f"valid_points={valid_pts} "
            f"dangles_deg={angles_deg} "
            f"dt={dt}"
        )

        if loss_raw < best["loss"]:
            best["loss"] = loss_raw
            best["raw_loss"] = loss_raw
            best["R"] = R.copy()
            best["t"] = t.copy()
            best["angles_deg"] = angles_deg.copy()
            best["valid_points"] = valid_pts
            best["per_class_info"] = info_cls

            print("   >>> nuevo mejor")
            print(f"   >>> best_loss={best['loss']:.6f}")
            print(f"   >>> best_per_class_info={best['per_class_info']}")

    return best

# VISUALIZACIÓN

def make_overlay(image_path, rgb_seg, points_xyz, lidar_labels, R, t, K, out_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {image_path}")

    uv, z_cam = project_points(points_xyz, R, t, K)

    vis = img.copy()
    H, W = rgb_seg.shape

    # colores simples por clase (solo para ver algo)
    rng = np.random.default_rng(1234)
    unique_cls = np.unique(lidar_labels)
    color_map = {
        int(c): tuple(int(x) for x in rng.integers(50, 255, size=3))
        for c in unique_cls
    }

    for (u, v), z, cls in zip(uv, z_cam, lidar_labels):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        if z <= 0:
            continue

        x = int(round(u))
        y = int(round(v))

        if 0 <= x < W and 0 <= y < H:
            color = color_map.get(int(cls), (0, 255, 0))
            cv2.circle(vis, (x, y), 3, color, -1)

    cv2.imwrite(out_path, vis)
    print(f"Guardado overlay: {out_path}")


# MAIN

def main():
    calib_path = "130/00000-000130-calib.json"
    image_path = "130/00000-000130-image.png"
    rgb_seg_path = "130/00000-000130-pred.png"
    points_path = "130/00000-000130-points.bin"
    labels_path = "130/00000-000130-pred.label"

    # Carga
    R0, t0, K, image_size = load_calib(calib_path)
    rgb_seg = load_rgb_seg(rgb_seg_path)
    points_xyz, _ = load_lidar_points(points_path)
    lidar_labels = load_lidar_labels(labels_path)

    print("rgb_seg shape:", rgb_seg.shape)
    print("points_xyz shape:", points_xyz.shape)
    print("lidar_labels shape:", lidar_labels.shape)
    print("image_size json:", image_size)

    if len(points_xyz) != len(lidar_labels):
        raise ValueError("El número de puntos LiDAR no coincide con el número de etiquetas")

    ignore_labels = set()

    class_ids = [12,14,17]

    # Búsqueda
    best = search_best_calibration(
            points_xyz=points_xyz,
            lidar_labels=lidar_labels,
            rgb_seg=rgb_seg,
            R0=R0,
            t0=t0,
            K=K,
            N=400,
            rot_range_deg=2.0,
            trans_range=0.05,
            ignore_labels=ignore_labels,
            class_ids=class_ids,
            seed=42
)

    print("\n=== MEJOR RESULTADO ===")
    print("best loss:", best["loss"])
    print("best valid_points:", best["valid_points"])
    print("best dangles_deg:", best["angles_deg"])
    print("best R:\n", best["R"])
    print("best t:\n", best["t"])
    print("best per_class_info:", best["per_class_info"])

    # Guarda overlay con la mejor calibración
    make_overlay(
        image_path=image_path,
        rgb_seg=rgb_seg,
        points_xyz=points_xyz,
        lidar_labels=lidar_labels,
        R=best["R"],
        t=best["t"],
        K=K,
        out_path="overlay_chamfer_unidireciconal_N400_1.png"
    )
    


    # Guarda la mejor calibración
    out = {
        "best_loss": float(best["loss"]),
        "best_valid_points": int(best["valid_points"]),
        "best_R": best["R"].tolist(),
        "best_t": best["t"].tolist(),
        "delta_angles_deg": best["angles_deg"].tolist()
    }

    with open("best_calibration_chamfer_unidireciconal_N400_1.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Guardado: best_calibration.json")
    

if __name__ == "__main__":
    main()