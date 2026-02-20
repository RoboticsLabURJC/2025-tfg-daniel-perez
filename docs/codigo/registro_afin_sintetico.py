import cv2
import numpy as np

# -------------------------
# Config
# -------------------------
BIG_PATH = "pattern_2.png"

OUT_BIG_WITH_PATCH = "big_with_patch4.png"
OUT_PATCH = "patch4.png"
OUT_SUBIMAGEN = "sub_imagen4.png"

# 1) Recorte original dentro de la grande
x0, y0 = 350, 280
w, h = 480, 560

# 2) Transformación aplicada
angle_deg = 37.0
scale = 1.3

# 3) Punto donde queremos anclar el centro del parche
anchor_cx, anchor_cy = 620, 520

# -------------------------
# Cargar imagen grande
# -------------------------
big = cv2.imread(BIG_PATH, cv2.IMREAD_COLOR)
if big is None:
    raise FileNotFoundError(f"No pude leer {BIG_PATH}")

H, W = big.shape[:2]

# -------------------------
# Sacar subimagen original
# -------------------------
patch = big[y0:y0+h, x0:x0+w].copy()
cv2.imwrite(OUT_PATCH, patch)

# -------------------------
# Crear transformación afín
# -------------------------
pcx, pcy = w / 2.0, h / 2.0
M = cv2.getRotationMatrix2D((pcx, pcy), angle_deg, scale)

# Ajustar traslación para anclar el centro
M[0, 2] += (anchor_cx - pcx)
M[1, 2] += (anchor_cy - pcy)

# -------------------------
# Aplicar transformación sobre canvas grande
# -------------------------
warped_full = cv2.warpAffine(
    patch,
    M,
    (W, H),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0),
)

# -------------------------
# Obtener bounding box real del parche transformado
# -------------------------
gray = cv2.cvtColor(warped_full, cv2.COLOR_BGR2GRAY)
coords = cv2.findNonZero((gray > 10).astype(np.uint8))

x, y, ww, hh = cv2.boundingRect(coords)

sub_imagen = warped_full[y:y+hh, x:x+ww].copy()
cv2.imwrite(OUT_SUBIMAGEN, sub_imagen)

# -------------------------
# Insertar parche transformado en la imagen grande
# -------------------------
mask = gray > 10
big_with_patch = big.copy()
big_with_patch[mask] = warped_full[mask]
cv2.imwrite(OUT_BIG_WITH_PATCH, big_with_patch)

# -------------------------
# Mostrar información
# -------------------------
print("=== Ground truth ===")
print(f"Recorte original (x0,y0,w,h): ({x0},{y0},{w},{h})")
print(f"Rotación (deg): {angle_deg}")
print(f"Escala: {scale}")
print(f"Anchor centro (cx,cy): ({anchor_cx},{anchor_cy})")
print("Matriz afín M (patch -> big):\n", M)

print("\nGenerado:")
print(" -", OUT_PATCH, "(subimagen original)")
print(" -", OUT_SUBIMAGEN, "(subimagen transformada ya recortada)")
print(" -", OUT_BIG_WITH_PATCH, "(imagen grande con el parche insertado)")
