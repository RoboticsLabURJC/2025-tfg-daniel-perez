import cv2
import numpy as np

# ==============================
# Configuración
# ==============================

PATCH_PATH = "sub_imagen4.png"      # tu parche pequeño (puede tener negro)
BIG_PATH   = "pattern_2.png"       # imagen grande

OUT_OVERLAY = "overlay_mask4.png"
OUT_MATCHES = "matches_inliers_mask4.png"


def affine_to_params(M):
    """
    Convierte una matriz afín 2x3 en:
    - escala
    - ángulo (grados)
    - traslación (tx, ty)
    """
    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = np.sqrt(a * a + c * c)
    angle = np.degrees(np.arctan2(c, a))
    return scale, angle, tx, ty


def make_nonblack_mask(img_bgr, thresh=10, morph=True):
    """
    Máscara booleana/uint8 donde img no es negro (o casi negro).
    thresh: umbral en [0..255] sobre escala de grises.
    morph: aplica limpieza morfológica para quitar ruido.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > thresh).astype(np.uint8) * 255

    if morph:
        # Quita puntitos y suaviza bordes (ajusta si lo necesitas)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    return mask


def main():
    patch = cv2.imread(PATCH_PATH, cv2.IMREAD_COLOR)
    big   = cv2.imread(BIG_PATH, cv2.IMREAD_COLOR)

    if patch is None:
        raise FileNotFoundError(f"No pude leer {PATCH_PATH}")
    if big is None:
        raise FileNotFoundError(f"No pude leer {BIG_PATH}")

    # ==============================
    # 1) Crear máscara para ignorar negro en PATCH
    # ==============================
    mask_patch = make_nonblack_mask(patch, thresh=10, morph=True)

    # (Opcional) guarda la máscara para comprobarla visualmente
    cv2.imwrite("mask_patch.png", mask_patch)

    # ==============================
    # 2) Detectar características ORB (con máscara en patch)
    # ==============================
    orb = cv2.ORB_create(8000)

    kp_patch, des_patch = orb.detectAndCompute(patch, mask_patch)
    kp_big, des_big     = orb.detectAndCompute(big, None)

    if des_patch is None or des_big is None or len(kp_patch) < 10 or len(kp_big) < 10:
        raise RuntimeError(
            "No hay suficientes features. Prueba:\n"
            "- aumentar tamaño del patch\n"
            "- bajar el umbral de máscara (thresh)\n"
            "- añadir más textura\n"
        )

    # ==============================
    # 3) Emparejamiento de descriptores
    # ==============================
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_patch, des_big)
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:1500]

    pts_patch = np.float32([kp_patch[m.queryIdx].pt for m in matches])
    pts_big   = np.float32([kp_big[m.trainIdx].pt for m in matches])

    # ==============================
    # 4) Estimar transformación afín parcial con RANSAC
    # ==============================
    M_est, inliers = cv2.estimateAffinePartial2D(
        pts_patch,
        pts_big,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.99
    )

    if M_est is None or inliers is None:
        raise RuntimeError("No se pudo estimar la transformación (M_est None).")

    inliers_bool = inliers.ravel().astype(bool)
    inlier_count = int(inliers_bool.sum())

    scale, angle, tx, ty = affine_to_params(M_est)

    print("=== Transformación estimada (PATCH -> BIG) ===")
    print("M_est:\n", M_est)
    print(f"Inliers: {inlier_count} / {len(matches)}")
    print(f"Escala:  {scale:.4f}")
    print(f"Ángulo:  {angle:.2f} deg")
    print(f"Tx, Ty:  ({tx:.2f}, {ty:.2f})")

    # Inversa por si quieres comparar en el otro sentido
    M_inv = cv2.invertAffineTransform(M_est)
    s2, ang2, tx2, ty2 = affine_to_params(M_inv)
    print("\n=== Inversa (BIG -> PATCH) ===")
    print("M_inv:\n", M_inv)
    print(f"Escala:  {s2:.4f}")
    print(f"Ángulo:  {ang2:.2f} deg")
    print(f"Tx, Ty:  ({tx2:.2f}, {ty2:.2f})")

    # ==============================
    # 5) Overlay con contorno blanco visible
    # ==============================
    H, W = big.shape[:2]

    patch_proj = cv2.warpAffine(
        patch,
        M_est,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Máscara del parche proyectado
    mask_proj = make_nonblack_mask(patch_proj, thresh=10, morph=False)

    overlay = big.copy()

    # Mezcla solo en zona válida
    valid = mask_proj.astype(bool)
    overlay[valid] = cv2.addWeighted(big, 0.65, patch_proj, 0.35, 0)[valid]

    # -----------------------------
    # Dibujar contorno blanco
    # -----------------------------
    contours, _ = cv2.findContours(mask_proj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), thickness=3)

    cv2.imwrite(OUT_OVERLAY, overlay)
    # ==============================
    # 6) Matches (solo INLIERS)
    # ==============================
    inlier_matches = [m for m, keep in zip(matches, inliers_bool) if keep]
    draw_n = min(200, len(inlier_matches))

    matches_img = cv2.drawMatches(
        patch, kp_patch,
        big, kp_big,
        inlier_matches[:draw_n],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(OUT_MATCHES, matches_img)

    print("\nGuardado:")
    print(f" - mask_patch.png (máscara del patch)")
    print(f" - {OUT_OVERLAY} (overlay con máscara)")
    print(f" - {OUT_MATCHES} (matches inliers con máscara)")


if __name__ == "__main__":
    main()