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

def evaluate_candidate(
    points_xyz,
    lidar_labels,
    rgb_seg,
    R,
    t,
    K,
    class_ids=None,
    ignore_labels=None,
    min_class_pixels=30,
    min_class_points=30,
    class_count_power=0.0
):
    uv, z_cam = project_points(points_xyz, R, t, K)

    loss, valid_pts, info_cls = semantic_alignment_loss_chamfer_bidirectional(
        uv,
        z_cam,
        lidar_labels,
        rgb_seg,
        class_ids=class_ids,
        ignore_labels=ignore_labels,
        squared=True,
        normalize=True,
        class_average=True,
        min_class_pixels=min_class_pixels,
        min_class_points=min_class_points,
        class_count_power=class_count_power,
        bidir_alpha=1.0,
        bidir_beta=0.1
    )

    return loss, valid_pts, info_cls

def euler_grid_deg(max_abs_deg, step_deg):
    vals = np.arange(-max_abs_deg, max_abs_deg + 1e-12, step_deg, dtype=np.float64)
    return vals

def search_best_calibration_grid_coarse_fine(
    points_xyz,
    lidar_labels,
    rgb_seg,
    R0,
    t0,
    K,
    ignore_labels=None,
    class_ids=None,
    coarse_rot_deg=2.0,
    coarse_step_deg=0.5,
    fine_rot_deg=0.5,
    fine_step_deg=0.1,
    do_translation_fine=True,
    fine_trans_m=0.03,
    fine_trans_step=0.01,
    min_class_pixels=30,
    min_class_points=30,
    class_count_power=0.0
):
    if ignore_labels is None:
        ignore_labels = set()

    best = {
        "base_loss": None,
        "loss": np.inf,
        "R": None,
        "t": None,
        "angles_deg": None,
        "valid_points": 0,
        "per_class_info": {}
    }

    # --------------------------------------------------
    # 0) BASE
    # --------------------------------------------------
    base_loss, base_valid, base_info = evaluate_candidate(
        points_xyz=points_xyz,
        lidar_labels=lidar_labels,
        rgb_seg=rgb_seg,
        R=R0,
        t=t0,
        K=K,
        class_ids=class_ids,
        ignore_labels=ignore_labels,
        min_class_pixels=min_class_pixels,
        min_class_points=min_class_points,
        class_count_power=class_count_power
    )
    best["base_loss"] = base_loss
    best["loss"] = base_loss
    best["R"] = R0.copy()
    best["t"] = t0.copy()
    best["angles_deg"] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    best["valid_points"] = base_valid
    best["per_class_info"] = base_info

    print(f"[BASE] loss={base_loss:.6f} valid_points={base_valid}")
    print(f"[BASE] per_class_info={base_info}")

    # --------------------------------------------------
    # 1) COARSE ROTATION GRID
    # --------------------------------------------------
    coarse_vals = euler_grid_deg(coarse_rot_deg, coarse_step_deg)
    total_coarse = len(coarse_vals) ** 3
    idx = 0

    print(
        f"\n[COARSE GRID] rot in [-{coarse_rot_deg}, {coarse_rot_deg}] "
        f"step={coarse_step_deg} deg -> {total_coarse} candidates"
    )

    for rx in coarse_vals:
        for ry in coarse_vals:
            for rz in coarse_vals:
                idx += 1

                dR = rodrigues_from_euler(
                    np.deg2rad(rx),
                    np.deg2rad(ry),
                    np.deg2rad(rz)
                )

                R = dR @ R0
                t = t0.copy()

                loss, valid_pts, info_cls = evaluate_candidate(
                    points_xyz=points_xyz,
                    lidar_labels=lidar_labels,
                    rgb_seg=rgb_seg,
                    R=R,
                    t=t,
                    K=K,
                    class_ids=class_ids,
                    ignore_labels=ignore_labels,
                    min_class_pixels=min_class_pixels,
                    min_class_points=min_class_points,
                    class_count_power=class_count_power
                )

                print(
                    f"[COARSE {idx:04d}/{total_coarse}] "
                    f"loss={loss:.6f} valid_points={valid_pts} "
                    f"angles=({rx:.3f}, {ry:.3f}, {rz:.3f})"
                )

                if loss < best["loss"]:
                    best["loss"] = loss
                    best["R"] = R.copy()
                    best["t"] = t.copy()
                    best["angles_deg"] = np.array([rx, ry, rz], dtype=np.float64)
                    best["valid_points"] = valid_pts
                    best["per_class_info"] = info_cls

                    print("   >>> nuevo mejor COARSE")
                    print(f"   >>> best_loss={best['loss']:.6f}")
                    print(f"   >>> best_angles_deg={best['angles_deg']}")
                    print(f"   >>> best_t={best['t']}")

    coarse_best_angles = best["angles_deg"].copy()

    # --------------------------------------------------
    # 2) FINE ROTATION GRID around coarse best
    # --------------------------------------------------
    fine_offsets = euler_grid_deg(fine_rot_deg, fine_step_deg)
    total_fine = len(fine_offsets) ** 3
    idx = 0

    print(
        f"\n[FINE GRID ROT] around {coarse_best_angles} "
        f"window=±{fine_rot_deg} step={fine_step_deg} deg "
        f"-> {total_fine} candidates"
    )

    for drx in fine_offsets:
        for dry in fine_offsets:
            for drz in fine_offsets:
                idx += 1

                rx = coarse_best_angles[0] + drx
                ry = coarse_best_angles[1] + dry
                rz = coarse_best_angles[2] + drz

                dR = rodrigues_from_euler(
                    np.deg2rad(rx),
                    np.deg2rad(ry),
                    np.deg2rad(rz)
                )

                R = dR @ R0
                t = t0.copy()

                loss, valid_pts, info_cls = evaluate_candidate(
                    points_xyz=points_xyz,
                    lidar_labels=lidar_labels,
                    rgb_seg=rgb_seg,
                    R=R,
                    t=t,
                    K=K,
                    class_ids=class_ids,
                    ignore_labels=ignore_labels,
                    min_class_pixels=min_class_pixels,
                    min_class_points=min_class_points,
                    class_count_power=class_count_power
                )

                print(
                    f"[FINE-ROT {idx:04d}/{total_fine}] "
                    f"loss={loss:.6f} valid_points={valid_pts} "
                    f"angles=({rx:.3f}, {ry:.3f}, {rz:.3f})"
                )

                if loss < best["loss"]:
                    best["loss"] = loss
                    best["R"] = R.copy()
                    best["t"] = t.copy()
                    best["angles_deg"] = np.array([rx, ry, rz], dtype=np.float64)
                    best["valid_points"] = valid_pts
                    best["per_class_info"] = info_cls

                    print("   >>> nuevo mejor FINE-ROT")
                    print(f"   >>> best_loss={best['loss']:.6f}")
                    print(f"   >>> best_angles_deg={best['angles_deg']}")
                    print(f"   >>> best_t={best['t']}")

    # --------------------------------------------------
    # 3) FINE TRANSLATION GRID around current best
    # --------------------------------------------------
    if do_translation_fine:
        best_angles = best["angles_deg"].copy()
        trans_vals = np.arange(
            -fine_trans_m,
            fine_trans_m + 1e-12,
            fine_trans_step,
            dtype=np.float64
        )

        total_trans = len(trans_vals) ** 3
        idx = 0

        print(
            f"\n[FINE GRID TRANS] around t0 "
            f"window=±{fine_trans_m} step={fine_trans_step} m "
            f"-> {total_trans} candidates"
        )

        dR_best = rodrigues_from_euler(
            np.deg2rad(best_angles[0]),
            np.deg2rad(best_angles[1]),
            np.deg2rad(best_angles[2])
        )
        R_fixed = dR_best @ R0

        for dtx in trans_vals:
            for dty in trans_vals:
                for dtz in trans_vals:
                    idx += 1

                    t = t0 + np.array([dtx, dty, dtz], dtype=np.float64)

                    loss, valid_pts, info_cls = evaluate_candidate(
                        points_xyz=points_xyz,
                        lidar_labels=lidar_labels,
                        rgb_seg=rgb_seg,
                        R=R_fixed,
                        t=t,
                        K=K,
                        class_ids=class_ids,
                        ignore_labels=ignore_labels,
                        min_class_pixels=min_class_pixels,
                        min_class_points=min_class_points,
                        class_count_power=class_count_power
                    )

                    print(
                        f"[FINE-TRANS {idx:04d}/{total_trans}] "
                        f"loss={loss:.6f} valid_points={valid_pts} "
                        f"dt=({dtx:.3f}, {dty:.3f}, {dtz:.3f})"
                    )

                    if loss < best["loss"]:
                        best["loss"] = loss
                        best["R"] = R_fixed.copy()
                        best["t"] = t.copy()
                        best["angles_deg"] = best_angles.copy()
                        best["valid_points"] = valid_pts
                        best["per_class_info"] = info_cls

                        print("   >>> nuevo mejor FINE-TRANS")
                        print(f"   >>> best_loss={best['loss']:.6f}")
                        print(f"   >>> best_angles_deg={best['angles_deg']}")
                        print(f"   >>> best_t={best['t']}")

    return best

def semantic_alignment_loss_chamfer_bidirectional(
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
    class_count_power=0.0,
    bidir_alpha=1.0,
    bidir_beta=1.0
):
    """
    Chamfer bidireccional por clase.

    Dirección 1: LiDAR proyectado -> máscara RGB de la misma clase
    Dirección 2: máscara RGB de la misma clase -> puntos LiDAR proyectados de esa clase

    loss_clase = alpha * mean( LiDAR->RGB ) + beta * mean( RGB->LiDAR )

    Parámetros
    ----------
    bidir_alpha : float
        Peso del término LiDAR->RGB.
    bidir_beta : float
        Peso del término RGB->LiDAR.
    """
    if ignore_labels is None:
        ignore_labels = set()

    H, W = rgb_seg.shape

    if class_ids is None:
        class_ids = sorted(int(c) for c in np.unique(lidar_labels) if int(c) not in ignore_labels)
    else:
        class_ids = [int(c) for c in class_ids if int(c) not in ignore_labels]

    per_class_info = {}
    total_loss = 0.0
    total_count = 0

    for cls in class_ids:
        # ---------------------------------------
        # máscara 2D de la clase
        # ---------------------------------------
        rgb_mask = (rgb_seg == cls).astype(np.uint8)
        pixel_count_2d = int(rgb_mask.sum())

        if pixel_count_2d < min_class_pixels:
            continue

        # ---------------------------------------
        # puntos LiDAR proyectados válidos de la clase
        # ---------------------------------------
        cls_mask = (lidar_labels == cls)

        proj_pixels = []
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

            proj_pixels.append((x, y))

        point_count_3d = len(proj_pixels)
        if point_count_3d < min_class_points:
            continue

        # ---------------------------------------
        # Dirección 1: LiDAR -> RGB
        # DT con ceros en la máscara RGB
        # ---------------------------------------
        inv_rgb_mask = 1 - rgb_mask
        dt_lidar_to_rgb = cv2.distanceTransform(
            inv_rgb_mask,
            distanceType=cv2.DIST_L2,
            maskSize=3
        )

        loss_lidar_to_rgb_sum = 0.0
        for x, y in proj_pixels:
            d = float(dt_lidar_to_rgb[y, x])
            if squared:
                d = d * d
            loss_lidar_to_rgb_sum += d

        loss_lidar_to_rgb_mean = loss_lidar_to_rgb_sum / point_count_3d

        # ---------------------------------------
        # Dirección 2: RGB -> LiDAR
        # construimos imagen binaria de puntos proyectados
        # y calculamos DT hacia esos puntos
        # ---------------------------------------
        proj_mask = np.zeros((H, W), dtype=np.uint8)
        for x, y in proj_pixels:
            proj_mask[y, x] = 1

        inv_proj_mask = 1 - proj_mask
        dt_rgb_to_lidar = cv2.distanceTransform(
            inv_proj_mask,
            distanceType=cv2.DIST_L2,
            maskSize=3
        )

        ys, xs = np.where(rgb_mask > 0)
        if len(xs) == 0:
            continue

        d_rgb = dt_rgb_to_lidar[ys, xs].astype(np.float64)
        if squared:
            d_rgb = d_rgb * d_rgb

        loss_rgb_to_lidar_sum = float(d_rgb.sum())
        loss_rgb_to_lidar_mean = loss_rgb_to_lidar_sum / len(d_rgb)

        # ---------------------------------------
        # combinación bidireccional
        # ---------------------------------------
        cls_mean = (
            float(bidir_alpha) * loss_lidar_to_rgb_mean +
            float(bidir_beta) * loss_rgb_to_lidar_mean
        )

        per_class_info[cls] = {
            "loss_mean": cls_mean,
            "loss_lidar_to_rgb_mean": loss_lidar_to_rgb_mean,
            "loss_rgb_to_lidar_mean": loss_rgb_to_lidar_mean,
            "loss_lidar_to_rgb_sum": loss_lidar_to_rgb_sum,
            "loss_rgb_to_lidar_sum": loss_rgb_to_lidar_sum,
            "count": point_count_3d,
            "pixel_count_2d": pixel_count_2d
        }

        total_loss += cls_mean
        total_count += point_count_3d

    if len(per_class_info) == 0:
        return np.inf, 0, {}

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

    else:
        if total_count == 0:
            return np.inf, 0, {}
        if normalize:
            total_loss = total_loss / total_count

    return total_loss, total_count, per_class_info

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


    # Búsqueda
    best = search_best_calibration_grid_coarse_fine(
        points_xyz=points_xyz,
        lidar_labels=lidar_labels,
        rgb_seg=rgb_seg,
        R0=R0,
        t0=t0,
        K=K,
        ignore_labels=set(),
        class_ids=[12, 14, 17],
        coarse_rot_deg=2.0,
        coarse_step_deg=1.0,
        fine_rot_deg=0.5,
        fine_step_deg=0.25,
        do_translation_fine=True,
        fine_trans_m=0.02,
        fine_trans_step=0.01,
        min_class_pixels=30,
        min_class_points=30,
        class_count_power=0.0
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
        out_path="overlay_chamfer_bidireciconal_grid.png"
    )
    


    # Guarda la mejor calibración
    out = {
        "base_loss": float(best["base_loss"]),
        "best_loss": float(best["loss"]),
        "best_valid_points": int(best["valid_points"]),
        "best_R": best["R"].tolist(),
        "best_t": best["t"].tolist(),
        "delta_angles_deg": best["angles_deg"].tolist()
    }

    with open("best_calibration_chamfer_bidireciconal_grid.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Guardado: best_calibration.json")
    print("t0:", t0)
    print("best t:", best["t"])
    print("delta t:", best["t"] - t0)
        

if __name__ == "__main__":
    main()