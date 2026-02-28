import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

H, W = 120, 120

red_1 = np.random.uniform(0.10, 0.55, (H, W))
nir_1 = np.random.uniform(0.35, 0.95, (H, W))

red_2 = red_1 + np.random.uniform(0.00, 0.10, (H, W))
nir_2 = nir_1 - np.random.uniform(0.00, 0.22, (H, W))

mask_stress = np.zeros((H, W), dtype=bool)
mask_stress[35:80, 40:95] = True
nir_2[mask_stress] = nir_2[mask_stress] * 0.65
red_2[mask_stress] = red_2[mask_stress] * 1.10

def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 1e-6)

ndvi_1 = calculate_ndvi(nir_1, red_1)
ndvi_2 = calculate_ndvi(nir_2, red_2)

delta_ndvi = ndvi_2 - ndvi_1

mu = float(delta_ndvi.mean())
sigma = float(delta_ndvi.std() + 1e-6)
z = (delta_ndvi - mu) / sigma

anomaly = (z < -2).astype(np.uint8)

risk = np.zeros_like(ndvi_2, dtype=np.uint8)

risk[(ndvi_2 < 0.30) | (delta_ndvi < -0.15) | (anomaly == 1)] = 2

risk[((ndvi_2 >= 0.30) & (ndvi_2 < 0.60)) | ((delta_ndvi >= -0.15) & (delta_ndvi < -0.05))] = np.maximum(
    risk[((ndvi_2 >= 0.30) & (ndvi_2 < 0.60)) | ((delta_ndvi >= -0.15) & (delta_ndvi < -0.05))],
    1
)

def save_map(data, title, filename, cmap, vmin=None, vmax=None):
    plt.figure(figsize=(6, 5))
    plt.title(title)
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.title("NDVI (Before)")
im1 = plt.imshow(ndvi_1, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im1, fraction=0.046, pad=0.04)

plt.subplot(1, 3, 2)
plt.title("NDVI (After)")
im2 = plt.imshow(ndvi_2, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im2, fraction=0.046, pad=0.04)

plt.subplot(1, 3, 3)
plt.title("ΔNDVI (Change)")
im3 = plt.imshow(delta_ndvi, cmap="coolwarm", vmin=-0.5, vmax=0.5)
plt.colorbar(im3, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("01_ndvi_temporal.png", dpi=220)
plt.close()

save_map(
    anomaly,
    "AI: Аномалия (қатты нашарлау зоналары)",
    "02_ai_anomaly.png",
    cmap="gray",
    vmin=0,
    vmax=1
)

save_map(
    risk,
    "AI: Денсаулық зоналары (0-қалыпты,1-ескерту,2-қауіпті)",
    "03_ai_risk_zones.png",
    cmap="RdYlGn_r",
    vmin=0,
    vmax=2
)

print("Дайын ✅ Файлдар сақталды:")
print(" - 01_ndvi_temporal.png")
print(" - 02_ai_anomaly.png")
print(" - 03_ai_risk_zones.png")
