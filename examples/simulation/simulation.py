"""
Simulation validation for pygenoscape

Generates:
- Isolation-by-distance (IBD) scenario
- Barrier-to-gene-flow scenario

Used to produce Figure 2 in:
Davinack & Seaberg (2026) Bioinformatics Advances
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.spatial.distance import pdist, squareform
from numpy.linalg import eigh

# ----------------------------
# Reproducibility
# ----------------------------
np.random.seed(42)

# ----------------------------
# PCoA implementation
# ----------------------------
def pcoa(distance_matrix):
    """
    Perform Principal Coordinates Analysis (PCoA) on a symmetric distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Square symmetric distance matrix.

    Returns
    -------
    coordinates : np.ndarray
        Principal coordinates.
    eigenvalues : np.ndarray
        Eigenvalues corresponding to the coordinates.
    """
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]

    # Double-center the squared distance matrix
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    # Eigen decomposition
    eigvals, eigvecs = eigh(B)

    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep only positive eigenvalues
    positive = eigvals > 1e-10
    eigvals = eigvals[positive]
    eigvecs = eigvecs[:, positive]

    coordinates = eigvecs * np.sqrt(eigvals)
    return coordinates, eigvals


# ----------------------------
# Simulate coordinates
# ----------------------------
def simulate_coordinates(n_points=60):
    """
    Simulate random sample coordinates in a 2D unit square.
    """
    x = np.random.uniform(0, 1, n_points)
    y = np.random.uniform(0, 1, n_points)
    return np.column_stack((x, y))


# ----------------------------
# Distance scenarios
# ----------------------------
def simulate_ibd_distance_matrix(coords, noise_sd=0.03):
    """
    Simulate an isolation-by-distance (IBD) genetic distance matrix.
    Genetic distance increases with Euclidean geographic distance plus noise.
    """
    geo_dist = squareform(pdist(coords, metric="euclidean"))
    noise = np.random.normal(0, noise_sd, size=geo_dist.shape)

    D = geo_dist + noise
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0)

    # Ensure no negative distances
    D[D < 0] = 0
    return D


def simulate_barrier_distance_matrix(coords, barrier_x=0.5, noise_sd=0.03, barrier_strength=0.5):
    """
    Simulate a genetic distance matrix with isolation-by-distance plus
    a barrier to gene flow at x = barrier_x.
    """
    geo_dist = squareform(pdist(coords, metric="euclidean"))
    noise = np.random.normal(0, noise_sd, size=geo_dist.shape)

    D = geo_dist + noise

    # Add a barrier penalty when samples are on opposite sides
    x = coords[:, 0]
    left = x < barrier_x
    right = x >= barrier_x

    for i in range(len(coords)):
        for j in range(len(coords)):
            if (left[i] and right[j]) or (right[i] and left[j]):
                D[i, j] += barrier_strength

    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0)
    D[D < 0] = 0
    return D


# ----------------------------
# Interpolation helper
# ----------------------------
def interpolate_surface(coords, values, grid_size=200, rbf_function="multiquadric", smooth=0.05):
    """
    Interpolate scalar values over a regular 2D grid using RBF interpolation.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size),
        np.linspace(y.min(), y.max(), grid_size)
    )

    rbf = Rbf(x, y, values, function=rbf_function, smooth=smooth)
    grid_z = rbf(grid_x, grid_y)

    return grid_x, grid_y, grid_z


# ----------------------------
# Generate simulated data
# ----------------------------
coords = simulate_coordinates(n_points=60)

# Scenario 1: Isolation by distance
D_ibd = simulate_ibd_distance_matrix(coords, noise_sd=0.03)
pcoa_ibd, eig_ibd = pcoa(D_ibd)
pc1_ibd = pcoa_ibd[:, 0]

# Scenario 2: Barrier to gene flow
D_barrier = simulate_barrier_distance_matrix(
    coords,
    barrier_x=0.5,
    noise_sd=0.03,
    barrier_strength=0.5
)
pcoa_barrier, eig_barrier = pcoa(D_barrier)
pc1_barrier = pcoa_barrier[:, 0]

# Interpolate surfaces
gx_ibd, gy_ibd, gz_ibd = interpolate_surface(coords, pc1_ibd, grid_size=200, smooth=0.05)
gx_bar, gy_bar, gz_bar = interpolate_surface(coords, pc1_barrier, grid_size=200, smooth=0.05)

# ----------------------------
# Shared color scale across panels
# ----------------------------
vmin = min(gz_ibd.min(), gz_bar.min())
vmax = max(gz_ibd.max(), gz_bar.max())

# ----------------------------
# Plot
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

# Panel A: IBD
im1 = axes[0].contourf(
    gx_ibd, gy_ibd, gz_ibd,
    levels=100,
    vmin=vmin,
    vmax=vmax
)
axes[0].scatter(
    coords[:, 0], coords[:, 1],
    s=35,
    edgecolor="black",
    linewidth=0.6
)
axes[0].set_title("A. Simulated isolation-by-distance")
axes[0].set_xlabel("X coordinate")
axes[0].set_ylabel("Y coordinate")
axes[0].text(
    0.02, 0.95, "Continuous gradient",
    transform=axes[0].transAxes,
    ha="left", va="top"
)
cbar1 = fig.colorbar(im1, ax=axes[0])
cbar1.set_label("Interpolated genetic variation (PCoA1)")

# Panel B: Barrier
im2 = axes[1].contourf(
    gx_bar, gy_bar, gz_bar,
    levels=100,
    vmin=vmin,
    vmax=vmax
)
axes[1].scatter(
    coords[:, 0], coords[:, 1],
    s=35,
    edgecolor="black",
    linewidth=0.6
)
axes[1].axvline(0.5, linestyle="--", linewidth=2, color="black")
axes[1].set_title("B. Simulated barrier to gene flow")
axes[1].set_xlabel("X coordinate")
axes[1].set_ylabel("Y coordinate")
axes[1].text(
    0.02, 0.95, "Barrier detected",
    transform=axes[1].transAxes,
    ha="left", va="top"
)
cbar2 = fig.colorbar(im2, ax=axes[1])
cbar2.set_label("Interpolated genetic variation (PCoA1)")

# Save figure
plt.savefig("pygenoscape_simulation_validation.png", dpi=300, bbox_inches="tight")
plt.show()