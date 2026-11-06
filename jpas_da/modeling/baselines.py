"""Linear baselines at the VAE's latent size, scored exactly like the VAE.

* **Weighted PCA on g** (``wpca``): the rank-``d`` model ``g ~ mu + A V^T`` that
  minimises ``sum m_j (g_j - g_hat_j)^2 / w_j^2`` -- the same chi2 the VAE's
  Gaussian likelihood minimises, with the same ``w``. Fitted by alternating
  weighted least squares (per-object coefficients, then per-band mean and
  components together), starting from the unweighted PCA.
* **PCA in asinh space** (``asinh_pca``): the textbook route with no noise model.
  ``x = asinh(f / b_j)`` with ``b_j`` the median error per band on the
  training tiles (:mod:`jpas_da.asinh`), masked cells filled with the training
  band mean, ordinary PCA, then the reconstruction is mapped back to flux, then
  to ``g`` and scored with the same chi2.

Both fit on the training tiles of a rotation and report on its validation and
test tiles. Neither gets ``log r_bb`` as an extra input, unlike the conditional
VAE; the asinh PCA does see brightness implicitly, through the flux level.

Run with ``python -m jpas_da.modeling.baselines`` (all rotations, sizes 4/8/16).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import torch
import typer

from jpas_da import asinh
from jpas_da.config import REPORTS_DIR
from jpas_da.features import MODEL_DATA_FILE
from jpas_da.modeling import data as model_data
from jpas_da.modeling import evaluate
from jpas_da.modeling.losses import noise_variance

app = typer.Typer()

BASELINES_FILE = REPORTS_DIR / "runs" / "baselines.json"


def weights_for(arrays: dict, index: np.ndarray, eps: float, common_mode: bool) -> np.ndarray:
    """``w^2`` of the VAE's noise model for the given rows (0 on masked cells)."""
    w2 = noise_variance(
        torch.from_numpy(arrays["g"][index].astype(np.float64)),
        torch.from_numpy(arrays["v"][index].astype(np.float64)),
        torch.from_numpy(arrays["mask"][index].astype(np.float64)),
        torch.from_numpy(arrays["sig_logr"][index].astype(np.float64)),
        eps,
        common_mode,
    )
    return w2.numpy()


def _inverse_weights(w2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 1.0 / np.maximum(w2, 1e-12), 0.0)


def _solve_coefficients(y: np.ndarray, weights: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Per-object weighted least squares for ``y ~ A V^T``; ``V`` is ``(bands, d)``."""
    d = components.shape[1]
    gram = np.einsum("nj,jk,jl->nkl", weights, components, components) + 1e-8 * np.eye(d)
    rhs = np.einsum("nj,jk,nj->nk", weights, components, y)
    return np.linalg.solve(gram, rhs[..., None])[..., 0]


class WeightedPCA:
    """Rank-``d`` weighted low-rank model with a weighted mean."""

    def __init__(self, n_components: int, n_iter: int = 30) -> None:
        self.n_components = n_components
        self.n_iter = n_iter

    def fit(self, g: np.ndarray, w2: np.ndarray, mask: np.ndarray) -> WeightedPCA:
        weights = _inverse_weights(w2, mask)
        g = np.where(mask > 0, g, 0.0)
        self.mean = (weights * g).sum(0) / np.maximum(weights.sum(0), 1e-12)
        _, _, vt = np.linalg.svd(np.where(mask > 0, g - self.mean, 0.0), full_matrices=False)
        components = vt[: self.n_components].T.copy()
        d = self.n_components
        for _ in range(self.n_iter):
            coeffs = _solve_coefficients(g - self.mean, weights, components)
            # Mean and components together: per band, regress g_j on [1, A].
            design = np.concatenate([np.ones((len(g), 1)), coeffs], axis=1)
            gram = np.einsum("nj,nk,nl->jkl", weights, design, design) + 1e-8 * np.eye(d + 1)
            rhs = np.einsum("nj,nk,nj->jk", weights, design, g)
            theta = np.linalg.solve(gram, rhs[..., None])[..., 0]
            self.mean, components = theta[:, 0], theta[:, 1:]
        # Orthonormalise for a canonical basis; the spanned subspace is unchanged.
        self.components, _ = np.linalg.qr(components)
        return self

    def reconstruct(self, g: np.ndarray, w2: np.ndarray, mask: np.ndarray) -> np.ndarray:
        weights = _inverse_weights(w2, mask)
        y = np.where(mask > 0, g - self.mean, 0.0)
        coeffs = _solve_coefficients(y, weights, self.components)
        return self.mean + coeffs @ self.components.T


class AsinhPCA:
    """Ordinary PCA in asinh space, reconstruction mapped back to ``g``."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    @staticmethod
    def _flux_and_sigma(g, v, logr):
        r = np.exp(np.asarray(logr, np.float64))[:, None]
        return np.asarray(g, np.float64) * r, np.asarray(v, np.float64) * r, r

    def fit(self, g, v, mask, logr) -> AsinhPCA:
        flux, sigma, _ = self._flux_and_sigma(g, v, logr)
        self.scale = asinh.fit_band_scales(sigma, mask)
        x = asinh.transform(flux, self.scale)
        measured = mask > 0
        self.fill = np.array([x[measured[:, j], j].mean() for j in range(x.shape[1])])
        x = np.where(measured, x, self.fill)
        self.mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[: self.n_components].T
        return self

    def reconstruct(self, g, v, mask, logr) -> np.ndarray:
        flux, _, r = self._flux_and_sigma(g, v, logr)
        x = np.where(mask > 0, asinh.transform(flux, self.scale), self.fill)
        coeffs = (x - self.mean) @ self.components
        x_hat = self.mean + coeffs @ self.components.T
        return asinh.inverse(x_hat, self.scale) / r


def score(arrays, index, g_hat, w2) -> dict[str, object]:
    chi2, dof = evaluate.chi2_per_object(arrays["g"][index], g_hat, w2, arrays["mask"][index])
    return evaluate.reconstruction_report(
        chi2, dof, arrays["sig_logr"][index], arrays["tile_id"][index], arrays["quality"][index]
    )


def run_rotation(
    arrays: dict, rotation: int, dims: list[int], eps: float, common_mode: bool
) -> list[dict[str, object]]:
    parts = model_data.split_indices(arrays, rotation)
    tr = parts["train"]
    w2 = {
        role: weights_for(arrays, parts[role], eps, common_mode)
        for role in ("train", "val", "test")
    }
    rows = []
    for d in dims:
        wpca = WeightedPCA(d).fit(arrays["g"][tr], w2["train"], arrays["mask"][tr])
        apca = AsinhPCA(d).fit(
            arrays["g"][tr], arrays["v"][tr], arrays["mask"][tr], arrays["logr"][tr]
        )
        for name, model in (("wpca", wpca), ("asinh_pca", apca)):
            entry: dict[str, object] = {"method": name, "latent_dim": d, "rotation": rotation}
            for role in ("val", "test"):
                idx = parts[role]
                if name == "wpca":
                    g_hat = model.reconstruct(arrays["g"][idx], w2[role], arrays["mask"][idx])
                else:
                    g_hat = model.reconstruct(
                        arrays["g"][idx],
                        arrays["v"][idx],
                        arrays["mask"][idx],
                        arrays["logr"][idx],
                    )
                entry[role] = {"reconstruction": score(arrays, idx, g_hat, w2[role])}
            pooled = entry["test"]["reconstruction"]["overall"]["chi2_dof_pooled"]
            logger.info(f"rotation {rotation} {name} d={d}: test chi2/dof pooled {pooled:.3f}")
            rows.append(entry)
    return rows


@app.command()
def main(
    data_path: Annotated[Path, typer.Option("--data")] = MODEL_DATA_FILE,
    output_path: Annotated[Path, typer.Option("--output")] = BASELINES_FILE,
    dims: Annotated[str, typer.Option("--dims")] = "4,8,16",
    rotations: Annotated[str, typer.Option("--rotations")] = "0,1,2,3",
    eps: Annotated[float, typer.Option("--eps")] = 0.1,
    common_mode: Annotated[bool, typer.Option("--common-mode/--no-common-mode")] = True,
) -> list[dict[str, object]]:
    arrays = model_data.load(data_path)
    rows = []
    for k in (int(x) for x in rotations.split(",")):
        rows += run_rotation(arrays, k, [int(x) for x in dims.split(",")], eps, common_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"eps": eps, "common_mode": common_mode, "runs": rows}, indent=2)
    )
    logger.success(f"Wrote {output_path} ({len(rows)} baseline fits)")
    return rows


if __name__ == "__main__":
    app()
