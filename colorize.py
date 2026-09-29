import cv2
import numpy as np

# -----------------------------
# 1. Model file paths
# -----------------------------
prototxt = "models/colorization_deploy_v2.prototxt"
model = "models/colorization_release_v2.caffemodel"
points = "models/pts_in_hull.npy"

# -----------------------------
# 2. Load the AI model
# -----------------------------
print("Loading AI model...")

net = cv2.dnn.readNetFromCaffe(prototxt, model)

# -----------------------------
# 3. Load color points
# -----------------------------
pts = np.load(points)
pts = pts.transpose().reshape(2, 313, 1, 1)

# Add color points to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

net.getLayer(class8).blobs = [
    pts.astype(np.float32)
]

net.getLayer(conv8).blobs = [
    np.full([1, 313], 2.606, dtype=np.float32)
]

# -----------------------------
# 4. Read black-and-white image
# -----------------------------
image = cv2.imread("input/image1.jpg")

if image is None:
    print("ERROR: input/image1.jpg not found!")
    exit()

# -----------------------------
# 5. Convert image to LAB
# -----------------------------
scaled = image.astype(np.float32) / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Resize for the AI model
resized = cv2.resize(lab, (224, 224))

# Get L channel
L = cv2.split(resized)[0]

# Mean-centering
L -= 50

# -----------------------------
# 6. AI predicts colors
# -----------------------------
print("Colorizing image...")

net.setInput(cv2.dnn.blobFromImage(L))

ab = net.forward()[0].transpose((1, 2, 0))

# Resize predicted colors to original image size
ab = cv2.resize(
    ab,
    (image.shape[1], image.shape[0])
)

# -----------------------------
# 7. Combine L + predicted colors
# -----------------------------
L = cv2.split(lab)[0]

colorized = np.concatenate(
    (L[:, :, np.newaxis], ab),
    axis=2
)

# Convert LAB back to BGR
colorized = cv2.cvtColor(
    colorized,
    cv2.COLOR_LAB2BGR
)

# Keep values between 0 and 1
colorized = np.clip(colorized, 0, 1)

# Convert to normal image
colorized = (255 * colorized).astype(np.uint8)

# -----------------------------
# 8. Save the colorized image
# -----------------------------
cv2.imwrite(
    "output/colorized_image.jpg",
    colorized
)

print("Colorization completed!")
print("Saved to: output/colorized_image.jpg")
