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

def semantic_alignment_loss(
    uv,
    z_cam,
    lidar_labels,
    rgb_seg,
    ignore_labels=None,
    neighborhood=0
):
    """
    Loss muy simple:
    - Si el punto proyectado cae dentro de la imagen y la clase coincide -> 0
    - Si no coincide -> 1

    neighborhood=0 -> compara solo pixel exacto
    neighborhood=1 o 2 -> busca coincidencia en vecindad local

    ignore_labels: clases que no quieres usar
    """
    if ignore_labels is None:
        ignore_labels = set()

    H, W = rgb_seg.shape
    total = 0.0
    count = 0

    for (u, v), z, cls in zip(uv, z_cam, lidar_labels):
        if cls in ignore_labels:
            continue

        if not np.isfinite(u) or not np.isfinite(v):
            continue

        if z <= 0:
            continue

        x = int(round(u))
        y = int(round(v))

        if x < 0 or x >= W or y < 0 or y >= H:
            continue

        matched = False

        x0 = max(0, x - neighborhood)
        x1 = min(W, x + neighborhood + 1)
        y0 = max(0, y - neighborhood)
        y1 = min(H, y + neighborhood + 1)

        patch = rgb_seg[y0:y1, x0:x1]
        if np.any(patch == cls):
            matched = True

        total += 0.0 if matched else 1.0
        count += 1

    if count == 0:
        return np.inf, 0

    return total / count, count

# BÚSQUEDA TIPO "LOSS-GUIDED INIT"

def search_best_calibration(
    points_xyz,
    lidar_labels,
    rgb_seg,
    R0,
    t0,
    K,
    N=200,
    rot_range_deg=2.0,
    trans_range=0.05,
    ignore_labels=None,
    neighborhood=0,
    seed=0
):
    np.random.seed(seed)

    best = {
        "loss": np.inf,
        "R": None,
        "t": None,
        "angles_deg": None,
        "valid_points": 0
    }

    # probamos la calibración base
    uv0, z0 = project_points(points_xyz, R0, t0, K)
    loss0, valid0 = semantic_alignment_loss(
        uv0, z0, lidar_labels, rgb_seg,
        ignore_labels=ignore_labels,
        neighborhood=neighborhood
    )

    best["loss"] = loss0
    best["R"] = R0.copy()
    best["t"] = t0.copy()
    best["angles_deg"] = np.array([0.0, 0.0, 0.0])
    best["valid_points"] = valid0

    print(f"[BASE] loss={loss0:.6f} valid_points={valid0}")

    for i in range(N):
        dR, dt, angles_deg = random_perturbation(
            rot_range_deg=rot_range_deg,
            trans_range=trans_range
        )

        # Perturbación alrededor de la calibración inicial
        R = dR @ R0
        t = t0 + dt

        uv, z_cam = project_points(points_xyz, R, t, K)
        loss, valid_pts = semantic_alignment_loss(
            uv, z_cam, lidar_labels, rgb_seg,
            ignore_labels=ignore_labels,
            neighborhood=neighborhood
        )

        print(
            f"[{i+1:03d}/{N}] "
            f"loss={loss:.6f} valid_points={valid_pts} "
            f"dangles_deg={angles_deg} dt={dt}"
        )

        if loss < best["loss"]:
            best["loss"] = loss
            best["R"] = R.copy()
            best["t"] = t.copy()
            best["angles_deg"] = angles_deg.copy()
            best["valid_points"] = valid_pts

    return best

# VISUALIZACIÓN

def make_overlay(image_path, rgb_seg, points_xyz, lidar_labels, R, t, K, out_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {image_path}")

    uv, z_cam = project_points(points_xyz, R, t, K)

    vis = img.copy()
    H, W = rgb_seg.shape

    # colores simples por clase
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
        neighborhood=3,        
        seed=42
    )

    print("\n=== MEJOR RESULTADO ===")
    print("best loss:", best["loss"])
    print("best valid_points:", best["valid_points"])
    print("best dangles_deg:", best["angles_deg"])
    print("best R:\n", best["R"])
    print("best t:\n", best["t"])

    # Guarda overlay con la mejor calibración
    make_overlay(
        image_path=image_path,
        rgb_seg=rgb_seg,
        points_xyz=points_xyz,
        lidar_labels=lidar_labels,
        R=best["R"],
        t=best["t"],
        K=K,
        out_path="overlay_N100_V3.png"
    )
    #Guardar calibracion inicial
    make_overlay(
        image_path=image_path,
        rgb_seg=rgb_seg,
        points_xyz=points_xyz,
        lidar_labels=lidar_labels,
        R=R0,
        t=t0,
        K=K,
        out_path="overlay_original.png"
    )

    # Guarda la mejor calibración
    out = {
        "best_loss": float(best["loss"]),
        "best_valid_points": int(best["valid_points"]),
        "best_R": best["R"].tolist(),
        "best_t": best["t"].tolist(),
        "delta_angles_deg": best["angles_deg"].tolist()
    }

    with open("best_calibration_400_3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Guardado: best_calibration.json")


if __name__ == "__main__":

    main()
