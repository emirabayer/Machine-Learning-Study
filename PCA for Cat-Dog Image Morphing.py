import numpy as np
import matplotlib.pyplot as plt
import os

DATASET_PATH = "cat_dog_images"
MORPH_SOURCE_PATH = "face_morphing/source.png"
MORPH_TARGET_PATH = "face_morphing/target.png"

# load all 60x60 grayscale png images from folder
images = []
for filename in os.listdir(DATASET_PATH):
    # read png as numpy array for values 0-255
    img = plt.imread(os.path.join(DATASET_PATH, filename))
    images.append(img.flatten())

# load all cat and dog images (4000x3600 matrix)
cd = np.array(images)
print(f"Loaded dataset shape: {cd.shape}")  # should be (4000, 3600)

# mean center the data
mean = np.mean(cd, axis=0)
data = cd - mean



# q1.1
print("SVD calculation in progress")
u, sigma, vt = np.linalg.svd(data, full_matrices=False)
pcs = vt[:10]
total_variance = np.sum(sigma**2)
pve = [(sigma[i]**2) / total_variance for i in range(10)]

print("PVE for the first 10 PCs:")
for i, n in enumerate(pve):
    print(f"PC {i+1}: {n:.4f} ({n*100:.2f}%)")


# q1.2
plt.figure(figsize=(15, 6))
for i in range(10):
    img = pcs[i].reshape(60, 60)
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"PC {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# q1.3
source = plt.imread(MORPH_SOURCE_PATH)
target = plt.imread(MORPH_TARGET_PATH)

source = source.flatten() - mean
target = target.flatten() - mean

Fsource = source @ vt.T  # Shape: (3600,)
Ftarget = target @ vt.T  # Shape: (3600,)

t_values = np.arange(0, 1.1, 0.1)  # 0 to 1 in steps of 0.1
morphed_images = []
for t in t_values:
    F_morp = (1 - t) * Fsource + t * Ftarget
    X_morp = F_morp @ vt + mean  # reconstruct and add mean back
    morphed_images.append(X_morp.reshape(60, 60))

# display morphed images
fig, axes = plt.subplots(1, 11, figsize=(22, 2))
for i, ax in enumerate(axes):
    ax.imshow(morphed_images[i], cmap="gray")
    ax.axis("off")
    ax.set_title(f"t={t_values[i]:.1f}")
plt.suptitle("Morphing from Cat to Dog")
plt.tight_layout()
plt.show()