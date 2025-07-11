import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 1. SDF Concept Overview
# ============================================
# A Signed Distance Function (SDF) returns:
#   - Positive value: point is outside the shape
#   - Zero: point lies exactly on the shape's surface
#   - Negative value: point is inside the shape
#
# Each shape has its own SDF formula.

# ============================================
# 2. Define SDF for a Circle
# ============================================
def sdf_circle(x, y, cx, cy, r):
    """
    Returns the signed distance from point (x, y)
    to the boundary of a circle centered at (cx, cy) with radius r.
    """
    dx = x - cx
    dy = y - cy
    return np.sqrt(dx**2 + dy**2) - r

# ============================================
# 3. Image Setup
# ============================================
width, height = 100, 100
img = np.zeros((height, width), dtype=np.float32)

# Circle parameters
center_x = 50
center_y = 50
radius = 30

# ============================================
# 4. SDF Edge Classification Visualization
#    (gray = on edge, white = inside, black = outside)
# ============================================
for y in range(height):
    for x in range(width):
        value = sdf_circle(x, y, center_x, center_y, radius)
        if abs(value) < 0.1:
            img[y, x] = 127   # Gray → On the edge
        elif value < 0:
            img[y, x] = 255   # White → Inside the circle
        else:
            img[y, x] = 0     # Black → Outside the circle

plt.imshow(img, cmap='gray')
plt.title("SDF – Circle Edge Classification")
plt.axis("off")
plt.show()

# ============================================
# 5. Soft SDF Visualization (Gradient Effect)
# ============================================
# Use SDF values directly to simulate soft edges or glow
for y in range(height):
    for x in range(width):
        img[y, x] = sdf_circle(x, y, center_x, center_y, radius)

# Clip values to range [-5, 5] to reduce extreme distances
img = np.clip(img, -5, 5)

# Normalize to [0, 1] for visualization
img = (img + 5) / 10.0

plt.imshow(img, cmap='plasma')
plt.title("SDF – Soft Gradient Visualization")
plt.axis("off")
plt.show()

# ============================================
# 🔍 Notes
# ============================================

# Note 1: SDF can be computed graphically
# ----------------------------------------
# Instead of using mathematical formulas, SDF can also be approximated
# from binary images using image processing (e.g., distance transform).
# This is useful when the shape is not defined analytically but given as a raster image.

# Note 2: SDF Represents Shape and Distance – Not Light
# -----------------------------------------------------
# The SDF only encodes the distance to the surface of a shape.
# It does not include color, lighting, glow, or shading.
# If you want visual effects like glowing edges or realistic shadows,
# you need to apply additional techniques (e.g., shading based on distance).

# Note 3: SDF is useful in deep learning and geometry-aware models
# ----------------------------------------------------------------
# Since SDF values are continuous and differentiable,
# they are widely used in deep learning models for shape representation,
# geometry-based simulations, and neural implicit surfaces.
