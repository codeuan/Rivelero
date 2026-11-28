"""Sample distinct plant cells from a seeded two-dimensional Perlin density."""
import numpy as np


def sample_plants(shape, count, cell_size_m, scale_m, contrast, rng):
    """Return local row/column indices. Scale is gradient lattice spacing in metres."""
    if not 1 <= count <= np.prod(shape) or scale_m <= 0 or not 0 <= contrast <= 20:
        raise ValueError("Invalid plant count, Perlin spacing or density contrast.")
    # Random phase prevents the same cells being anchored at lattice intersections.
    y = np.arange(shape[0])[:, None] * cell_size_m[0] / scale_m + rng.random()
    x = np.arange(shape[1])[None, :] * cell_size_m[1] / scale_m + rng.random()
    iy, ix = y.astype(int), x.astype(int)
    # Compact indices keep memory proportional to the domain even for fine lattices.
    ys, yi = np.unique(np.stack([iy, iy+1]), return_inverse=True)
    xs, xi = np.unique(np.stack([ix, ix+1]), return_inverse=True)
    yi, xi = yi.reshape(2, -1, 1), xi.reshape(2, 1, -1)
    angles = rng.uniform(0, 2*np.pi, (len(ys), len(xs)))
    gy, gx = np.sin(angles), np.cos(angles)
    fy, fx = y-iy, x-ix

    def dot(dy, dx):
        index = (yi[dy], xi[dx])
        return gy[index]*(fy-dy) + gx[index]*(fx-dx)

    def fade(t):
        return t**3 * (t*(6*t-15)+10)

    u, v = fade(fx), fade(fy)
    field = (1-v)*((1-u)*dot(0, 0) + u*dot(0, 1)) + v*((1-u)*dot(1, 0) + u*dot(1, 1))
    weights = np.exp(contrast*(field-field.max())).ravel()
    cells = rng.choice(weights.size, count, replace=False, p=weights/weights.sum())
    return np.column_stack(np.unravel_index(cells, shape))
