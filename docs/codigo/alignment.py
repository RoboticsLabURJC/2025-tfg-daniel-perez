import cv2
import numpy as np

CAM_PATH = "salidafinal.png"
LIDAR_PATH = "lidar_projected_radius3.png"

def color_ids(bgr):
    b = bgr[:,:,0].astype(np.int32)
    g = bgr[:,:,1].astype(np.int32)
    r = bgr[:,:,2].astype(np.int32)
    return (b << 16) | (g << 8) | r

def boundaries_from_labels(bgr):
    """Mapa de bordes de regiones: 255 donde cambia la etiqueta (color) entre vecinos."""
    ids = color_ids(bgr)
    h, w = ids.shape
    edges = np.zeros((h, w), np.uint8)

    # Diferencias con vecino derecho y abajo
    edges[:, :-1] |= (ids[:, :-1] != ids[:, 1:]).astype(np.uint8) * 255
    edges[:-1, :] |= (ids[:-1, :] != ids[1:, :]).astype(np.uint8) * 255

    # Engordar un poco para dar señal a ECC
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    edges = cv2.dilate(edges, k, iterations=1)
    return edges

def dt_from_edges(edges_u8):
    inv = 255 - edges_u8
    dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dt = 1.0 / (1.0 + dt)
    dt = cv2.normalize(dt, None, 0.0, 1.0, cv2.NORM_MINMAX)
    dt = cv2.GaussianBlur(dt, (0,0), 1.5)
    return dt.astype(np.float32)

def bbox_from_mask(mask_u8, pad=20):
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return (0, 0, mask_u8.shape[1], mask_u8.shape[0])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(mask_u8.shape[1]-1, x1 + pad); y1 = min(mask_u8.shape[0]-1, y1 + pad)
    return (x0, y0, x1+1, y1+1)

def init_phasecorr(template_f32, input_f32, window_u8=None):
    t = template_f32.copy()
    i = input_f32.copy()
    if window_u8 is not None:
        w = (window_u8.astype(np.float32)/255.0)
        w = cv2.GaussianBlur(w, (0,0), 3)
        t *= w
        i *= w
    (dx, dy), resp = cv2.phaseCorrelate(t, i)
    return dx, dy, resp

def ecc_align(template_f32, input_f32, input_mask_u8=None,
              motion=cv2.MOTION_TRANSLATION, levels=4, iters=400, eps=1e-6, warp_init=None):

    warp = np.eye(3, 3, dtype=np.float32) if motion == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)

    if warp_init is not None:
        warp[:] = warp_init

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    # --- IMPORTANTE: si warp_init está en full-res, hay que ESCALARLO al nivel más coarse ---
    coarse_lv = levels - 1
    coarse_scale = 1.0 / (2 ** coarse_lv)
    warp[0, 2] *= coarse_scale
    warp[1, 2] *= coarse_scale

    # Recorremos de coarse -> fine
    for lv in reversed(range(levels)):
        scale = 1.0 / (2 ** lv)
        t = cv2.resize(template_f32, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        i = cv2.resize(input_f32, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        m = None
        if input_mask_u8 is not None:
            m = cv2.resize(input_mask_u8, (i.shape[1], i.shape[0]), interpolation=cv2.INTER_NEAREST)

        try:
            cc, warp = cv2.findTransformECC(t, i, warp, motion, criteria, m)
            print(f"[ECC] level {lv} scale {scale:.4f} cc={cc:.4f}")
        except cv2.error as e:
            print(f"[ECC] level {lv} falló: {e}")
            break

        # Subimos el warp al siguiente nivel (más fino): traslación x2
        if lv > 0:
            warp[0, 2] *= 2.0
            warp[1, 2] *= 2.0

    return warp

def warp_seg(seg_bgr, warp, out_hw, motion):
    h, w = out_hw
    if motion == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(seg_bgr, warp, (w, h), flags=cv2.INTER_NEAREST)
    return cv2.warpAffine(seg_bgr, warp, (w, h), flags=cv2.INTER_NEAREST)

def main():
    cam = cv2.imread(CAM_PATH)
    lidar = cv2.imread(LIDAR_PATH)
    if cam is None or lidar is None:
        raise ValueError("No se pudieron cargar las imágenes.")

    # Asegura mismo tamaño (si ya están, no hace daño)
    lidar = cv2.resize(lidar, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Máscara de datos LiDAR (no-negro)
    lidar_ids = color_ids(lidar)
    lidar_mask = (lidar_ids != 0).astype(np.uint8) * 255  # negro puro como vacío

    # Recorta a zona donde hay LiDAR para garantizar solape
    x0, y0, x1, y1 = bbox_from_mask(lidar_mask, pad=30)
    cam_crop = cam[y0:y1, x0:x1]
    lidar_crop = lidar[y0:y1, x0:x1]
    lidar_mask_crop = lidar_mask[y0:y1, x0:x1]

    # (NUEVO) Rellenar un poco las "rayas" del LiDAR en la máscara
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    lidar_mask_crop = cv2.morphologyEx(lidar_mask_crop, cv2.MORPH_CLOSE, k_close, iterations=1)

    # Bordes por cambios de etiqueta (mejor que “máscara no-fondo”)
    cam_edges = boundaries_from_labels(cam_crop)
    lidar_edges = boundaries_from_labels(lidar_crop)

    cam_dt = dt_from_edges(cam_edges)
    lidar_dt = dt_from_edges(lidar_edges)

    # (NUEVO) Aplicar la MISMA máscara a ambas imágenes de registro (cam y lidar)
    w = (lidar_mask_crop.astype(np.float32) / 255.0)
    w = cv2.GaussianBlur(w, (0, 0), 5)   # suaviza el borde de la máscara
    cam_dt_w = cam_dt * w
    lidar_dt_w = lidar_dt * w

    # Init coarse por phase correlation (usar version enmascarada)
    dx, dy, resp = init_phasecorr(cam_dt_w, lidar_dt_w, window_u8=lidar_mask_crop)
    print("PhaseCorr init dx,dy,resp:", dx, dy, resp)

    motion = cv2.MOTION_TRANSLATION
    warp_init = np.array([[1, 0, dx],
                          [0, 1, dy]], dtype=np.float32)

    # ECC (usar version enmascarada)
    warp_local = ecc_align(cam_dt_w, lidar_dt_w, input_mask_u8=lidar_mask_crop,
                   motion=motion, levels=4, warp_init=warp_init)

    # Warp global (sin offsets)
    warp_global = warp_local.copy()

    aligned = warp_seg(lidar, warp_global, (cam.shape[0], cam.shape[1]), motion)

    # Debug
    cv2.imwrite("debug_cam_edges.png", cam_edges)
    cv2.imwrite("debug_lidar_edges.png", lidar_edges)
    cv2.imwrite("lidar_aligned_ecc.png", aligned)

    overlay = cv2.addWeighted(cam, 0.6, aligned, 0.4, 0)
    cv2.imwrite("debug_overlay.png", overlay)

    print("Guardado: lidar_aligned_ecc.png, debug_overlay.png, debug_*_edges.png")
    print("Warp global:\n", warp_global)

if __name__ == "__main__":
    main()