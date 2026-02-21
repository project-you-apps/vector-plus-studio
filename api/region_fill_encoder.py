"""
Region-Fill Encoder for Vector+ Studio.

Binary sign encoding: each embedding dimension gets a 64×64 region.
If embedding[i] > 0, the entire region is filled (4096 pixels ON).
If embedding[i] <= 0, the entire region stays OFF.

768 dims → rows 0-11, all 64 columns. Rows 12-63 are free for text/hippo.

This encoding preserves angular geometry (sign structure) making it
suitable for both search (Hamming) and associative recall (settle).
"""

import numpy as np


class RegionFillEncoderNomic768:
    """Region-fill encoder for 768-dim Nomic embeddings on a 4096×4096 lattice."""

    def __init__(self, n_dims=768, lattice_size=4096, region_size=64):
        self.n_dims = n_dims
        self.lattice_size = lattice_size
        self.region_size = region_size
        self.regions_per_row = lattice_size // region_size  # 64

        # Region-fill uses rows 0-11 (768 / 64 = 12 rows)
        self.num_rows_used = (n_dims + self.regions_per_row - 1) // self.regions_per_row

        print(f"RegionFillEncoder: {n_dims} dims, {self.num_rows_used} rows "
              f"(rows 0-{self.num_rows_used - 1}), rows {self.num_rows_used}-63 free")

    def encode(self, embedding):
        """
        Encode 768-dim embedding to 4096×4096 lattice pattern.

        Uses 4D→reshape layout matching standalone tests (test_cam_poseidon.py).
        grid[region_row, region_col, pixel_row, pixel_col] reshaped to (4096, 4096).
        Each region becomes a single pixel row spanning all 4096 columns.

        Args:
            embedding: np.ndarray of shape (768,) with float values

        Returns:
            np.ndarray: Shape (4096, 4096) with int8 values (0 or 1)
        """
        grid = np.zeros((self.regions_per_row, self.regions_per_row,
                         self.region_size, self.region_size), dtype=np.int8)

        for i in range(self.n_dims):
            if embedding[i] > 0:
                grid[i // self.regions_per_row, i % self.regions_per_row, :, :] = 1

        return grid.reshape(self.lattice_size, self.lattice_size)

    def decode(self, lattice):
        """
        Decode settled lattice pattern back to sign vector.

        Uses 4D reshape layout matching encode() — grid[r, c, pr, pc].
        Returns float values in {0.0, 1.0} where 1.0 means the original
        embedding dimension was positive. Compatible with Hamming comparison.

        Args:
            lattice: np.ndarray of shape (4096, 4096) or flattened

        Returns:
            np.ndarray: Shape (768,) with float values in [0, 1]
        """
        if lattice.ndim == 1:
            lattice = lattice.reshape(self.lattice_size, self.lattice_size)

        # Reshape to 4D matching encode layout, then threshold region means.
        # Do NOT binary-threshold pixels first; intermediate values (0.3, 0.4)
        # after physics settle carry information that affects borderline regions.
        grid = lattice.reshape(self.regions_per_row, self.regions_per_row,
                               self.region_size, self.region_size)
        embedding = np.zeros(self.n_dims, dtype=np.float32)

        for i in range(self.n_dims):
            region = grid[i // self.regions_per_row, i % self.regions_per_row, :, :]
            embedding[i] = 1.0 if np.mean(region) > 0.5 else 0.0

        return embedding

    def compute_sparsity(self, lattice):
        """Compute activation sparsity of lattice pattern."""
        active = np.sum(lattice > 0)
        return (active / lattice.size) * 100
