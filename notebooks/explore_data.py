import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import torch

    from jpas_da import asinh, features
    from jpas_da.bands import BAND_INDEX, BANDS, BAND_NAMES, BLOCK_ORDER, N_BANDS
    from jpas_da.config import INTERIM_DATA_DIR, MODELS_DIR, REPORTS_DIR
    from jpas_da.features import NORM_BANDS
    from jpas_da.modeling import data as model_data
    from jpas_da.modeling import evaluate, losses
    from jpas_da.modeling.vae import build_model

    return (
        BANDS,
        BAND_NAMES,
        BLOCK_ORDER,
        INTERIM_DATA_DIR,
        MODELS_DIR,
        N_BANDS,
        REPORTS_DIR,
        alt,
        asinh,
        build_model,
        evaluate,
        features,
        json,
        losses,
        model_data,
        mo,
        np,
        pd,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The data: from the SQL export to one training row

    This notebook follows one object end to end and then shows what a batch of them
    looks like, so that every number the VAE consumes can be traced back to a column
    in the export. It runs on the real files and needs `make data` first; the last
    section needs a checkpoint from `make train`.

    | artifact | built by | holds | used by |
    |---|---|---|---|
    | `data/raw/miniJpas_fluxes_errors_redshifts_full.csv` | the archive, via the SQL in `references/queries/` | 64,327 rows: 60 fluxes + 60 relative errors + metadata | `dataset.py` only |
    | `data/interim/objects.parquet` | `make interim` | one row per object: flux, **absolute** sigma, validity mask, metadata, `holdout`, `quality` | `features`, `splits`, `modeling.data` |
    | `data/interim/splits.parquet` | `make splits` | role of every object in each of the 4 tile rotations | analysis joins; training recomputes the roles |
    | `data/processed/vae_inputs.npz` | `make features` | `g`, `v`, `mask`, `logr`, `sig_logr` + identifiers | **`train.py`, `predict.py`, the baselines** |

    The structural fact to keep in mind: the VAE never sees a raw flux. It sees the
    **shape** `g = f / r_bb` and the **brightness** `log r_bb` as separate inputs, and
    the likelihood weights every band by `w = sqrt(v^2 + eps^2 + (g*sig_logr)^2)`.
    Sections 1–6 get there from the CSV; sections 7–10 describe what the model
    actually optimises; section 11 checks a trained checkpoint.

    Documentation: `references/getting-started.md`, `references/data_dictionary.md`,
    `references/pipeline-v2-plan.md`.
    """)
    return


@app.cell
def _(INTERIM_DATA_DIR, model_data, pd):
    arrays = model_data.load()  # the arrays training reads, with key validation
    objects = pd.read_parquet(INTERIM_DATA_DIR / "objects.parquet")
    splits = pd.read_parquet(INTERIM_DATA_DIR / "splits.parquet")
    return arrays, objects, splits


@app.cell
def _(arrays, mo, objects, np):
    # Every array is in canonical band order and aligned row by row: the same
    # object sits at the same index in all of them.
    assert (objects["object_id"].to_numpy() == arrays["object_id"]).all()

    g = arrays["g"]
    v = arrays["v"]
    mask = arrays["mask"]
    logr = arrays["logr"]
    sig_logr = arrays["sig_logr"]
    tile_id = arrays["tile_id"]
    object_id = arrays["object_id"]
    quality = arrays["quality"]
    holdout = arrays["holdout"]

    mo.md(
        f"""
        ## 1. The sample at a glance

        | | |
        |---|---|
        | objects | **{len(g):,}** |
        | channels per object | **{g.shape[1]}** (54 narrow + `uJAVA` + 4 broad + `J1007`) |
        | tiles | **{len(np.unique(tile_id))}** pointings: {", ".join(str(t) for t in np.unique(tile_id))} |
        | objects with all 60 errors measured | {int((mask == 1).all(axis=1).sum()):,} ({100 * (mask == 1).all(axis=1).mean():.2f}%) |
        | masked pixels | {int((mask == 0).sum()):,} ({100 * (mask == 0).mean():.2f}%) |
        | inference holdout (SDSS spectra) | {int(holdout.sum()):,} |
        | object id | `<tile_id>-<number>`, e.g. `{object_id[0]}` |

        The arrays are `float32` (`g`, `v`, `logr`, `sig_logr`) and `int8` (`mask`),
        `{g.nbytes / 2**20:.0f} MB` in total, loaded from
        `data/processed/vae_inputs.npz`.
        """
    )
    return g, holdout, logr, mask, object_id, quality, sig_logr, tile_id, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How to load it

    ```python
    from jpas_da.modeling.data import load, split_indices, model_inputs

    a = load()                                  # data/processed/vae_inputs.npz
    g, v, mask = a["g"], a["v"], a["mask"]      # (n_objects, 60)
    logr, sig_logr = a["logr"], a["sig_logr"]   # (n_objects,)

    parts = split_indices(a, rotation=0)        # train/val/test/holdout row indices
    blocks = model_inputs(a, ("g", "v", "mask"), ("logr",))
    blocks["narrow"].shape, blocks["other"].shape, blocks["cond"].shape
    # (64293, 3, 54), (64293, 18), (64293, 1)
    ```

    `load()` refuses a file missing any of `g, v, mask, logr, sig_logr, object_id,
    tile_id, holdout, quality`; `model_inputs` refuses an unknown channel or a
    condition set without `logr`. Band order is the single definition in
    `jpas_da.bands.BAND_NAMES`, so no column matching can drift.
    """)
    return


@app.cell
def _(BANDS, BLOCK_ORDER, alt, mo, np, pd):
    band_table = pd.DataFrame(
        [
            {
                "band": b.name,
                "block": b.block,
                "wavelength_nm": b.wavelength_aa / 10.0,
                "channel": i,
            }
            for i, b in enumerate(BANDS)
        ]
    )
    layout = (
        alt.Chart(band_table)
        .mark_tick(size=18, thickness=3)
        .encode(
            x=alt.X("wavelength_nm:Q", title="effective wavelength [nm]"),
            y=alt.Y("channel:Q", title="channel index", axis=alt.Axis(tickCount=8)),
            color=alt.Color("block:N", title="block", sort=list(BLOCK_ORDER)),
            tooltip=["band", "block", "wavelength_nm", "channel"],
        )
        .properties(
            height=280,
            title="channel index -> wavelength (54 narrow bands are the low-resolution spectrum)",
        )
    )
    narrow_w = band_table.loc[band_table["block"] == "narrow", "wavelength_nm"]
    counts = band_table["block"].value_counts().reindex(list(BLOCK_ORDER))
    mo.vstack(
        [
            mo.md(
                f"""
                ## 2. The 60 channels

                | block | n | members | role |
                |---|---|---|---|
                | narrow | 54 | `J0378` ... `J0910` | the low-resolution spectrum, 10 nm step (`J0378`->`J0390` is 12 nm) |
                | medium | 1 | `uJAVA` | 49.5 nm FWHM at 350 nm, the blue anchor |
                | broad | 4 | `uJPAS`, `gSDSS`, `rSDSS`, `iSDSS` | `rSDSS` is the detection band, far deeper than a narrow band |
                | highpass | 1 | `J1007` | the red edge |

                The narrow bands span {narrow_w.min():.0f}-{narrow_w.max():.0f} nm with a
                strictly increasing wavelength. Adjacent narrow bands overlap, which is
                why the encoder is a 1-D convolution along this axis rather than a dense
                layer. `wavelength_aa` is in Angstrom for all 60 bands; the filter names
                encode nanometres, hence the /10 here.
                """
            ),
            mo.hstack(
                [
                    mo.stat(value=f"{int(counts[b])}", label=f"{b}", bordered=True)
                    for b in counts.index
                ]
            ),
            layout,
        ]
    )
    return


@app.cell
def _(BAND_NAMES, INTERIM_DATA_DIR, mo, np, objects, pd):
    # The photometry as stored: flux, absolute sigma, mask, straight from the
    # interim table -- no transformation, so the numbers below are the raw claim.
    flux = objects[[f"flux_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float64)
    sigma = objects[[f"sigma_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float64)
    stored_mask = objects[[f"mask_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.int8)
    audit = {
        "rows": len(objects),
        "columns": objects.shape[1],
        "masked_pixels": int((stored_mask == 0).sum()),
        "masked_fraction": float((stored_mask == 0).mean()),
        "complete": int((stored_mask == 1).all(axis=1).sum()),
        "sigma_nan_matches_mask": bool(((stored_mask == 0) == np.isnan(sigma)).all()),
        "negative_flux": float((flux < 0).mean()),
        "zero_flux_masked": int(((flux == 0) & (stored_mask == 0)).sum()),
        "holdout": int(objects["holdout"].sum()),
        "quality": objects["quality"].value_counts().to_dict(),
        "sp_class": objects["spCLASSsdss"].value_counts().to_dict(),
        "zph": int(objects["zph"].notna().sum()),
        "zsp": int(objects["zsp"].notna().sum()),
    }
    per_tile = (
        pd.DataFrame({"tile_id": objects["tile_id"], "masked": (stored_mask == 0).any(axis=1)})
        .groupby("tile_id")["masked"]
        .mean()
    )
    tile_rows = (chr(10) + " " * 8).join(
        f"| {int(t)} | {100 * share:.2f}% |" for t, share in per_tile.items()
    )
    mo.md(
        f"""
        ## 3. What `dataset.py` repaired

        The export is a SQL dump, not an analysis-ready table. `dataset.py` is the only
        irreversible step, and everything below is measured on the files on disk:

        | property | value |
        |---|---|
        | objects x columns | {audit["rows"]:,} x {audit["columns"]} |
        | masked pixels | {audit["masked_pixels"]:,} ({100 * audit["masked_fraction"]:.3f}%) |
        | objects with all 60 measured | {audit["complete"]:,} |
        | `sigma` is NaN exactly where `mask == 0` | **{audit["sigma_nan_matches_mask"]}** |
        | pixels with negative flux | {100 * audit["negative_flux"]:.2f}% |
        | holdout (SDSS spectra) | {audit["holdout"]:,} |
        | quality classes | {", ".join(f"{k} {v:,}" for k, v in audit["quality"].items())} |
        | `spCLASSsdss` | {", ".join(f"{k} {v}" for k, v in audit["sp_class"].items())} |
        | `zph` / `zsp` usable | {audit["zph"]:,} / {audit["zsp"]:,} |

        Three repairs matter:

        1. **`FLUX_RELERR_AUTO` is signed and relative.** `sigma = |flux * relerr|`.
           Four columns of the archive are not absolute errors.
        2. **`relerr == 0` means "not computed", not "infinitely precise"** — 1.009% of
           pixels, unevenly spread over the pointings:

           | tile | objects with >= 1 masked channel |
           |---|---|
           {tile_rows}

           They become `sigma = NaN`, `mask = 0`. {audit["zero_flux_masked"]:,} of them
           have `flux == 0`; the rest keep a real flux with an unknown error, which is
           exactly why a mask exists instead of imputing.
        3. **Sentinels become NaN**: `zph = -9999` (813 rows), `zsp < 0` (193 rows), and
           `MAGrSDSS = 99` (902 rows). Left in place, they would enter a fit as values.

        `MAGrSDSS` and `angDist` are kept as metadata only: the first is exactly
        `26.1147 - 2.5*log10(rSDSS)`, the second is non-null exactly on SDSS-matched rows,
        so either would leak label availability into a feature.
        """
    )
    return audit, flux, sigma, stored_mask


@app.cell
def _(BAND_NAMES, alt, flux, mo, np, pd, stored_mask):
    measured_cells = stored_mask == 1
    measured_values = flux[measured_cells]

    def _hist(x, bins=80):
        counts, edges = np.histogram(x, bins=bins)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

    hist_flux = (
        alt.Chart(_hist(np.sign(measured_values) * np.log10(np.abs(measured_values) + 1.0)))
        .mark_bar()
        .encode(
            x=alt.X("left:Q", title="sign(f) * log10(|f| + 1)  [archive flux units]"),
            x2="right:Q",
            y=alt.Y("count:Q", title="pixels", scale=alt.Scale(type="log")),
        )
        .properties(height=180, width=430, title="raw flux")
    )
    negative_share = float((measured_values < 0).mean())
    flux_span = float(
        np.log10(np.abs(measured_values).max() / np.percentile(np.abs(measured_values), 1))
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 4. Why the flux cannot go into a network as it is

                * **{100 * negative_share:.1f}% of measured pixels have negative flux.** That is
                  noise, not astrophysics, but it is real data: a logarithm would need an
                  arbitrary floor or an imputation.
                * `|flux|` spans about **{flux_span:.0f} decades** from its 1st percentile to
                  its maximum, so `sum(f^2)` is dominated by a handful of pixels and a
                  Euclidean loss would fit those and ignore the rest.
                * The unit is not uJy: the zeropoint is 26.1147.
                """
            ),
            mo.hstack([hist_flux], justify="start"),
        ]
    )
    return


@app.cell
def _(BAND_NAMES, alt, arrays, asinh, flux, mask, mo, model_data, np, pd, v):
    # The asinh view, exactly as the PCA baseline builds it: b_j is fitted on the
    # training tiles of rotation 0, never on the whole catalogue.
    train_rows = model_data.split_indices(arrays, 0)["train"]
    b_j = asinh.fit_band_scales(flux[train_rows], mask[train_rows])
    x = asinh.transform(flux, b_j)
    sigma_x = asinh.error(flux, np.where(mask == 1, np.abs(flux), np.nan), b_j)

    ok = mask == 1
    residual = np.abs(asinh.inverse(x, b_j) - flux)[ok]
    rel = residual / np.maximum(np.abs(flux)[ok], 1e-12)
    comparison = pd.DataFrame(
        {
            "band": list(BAND_NAMES),
            "b_j": b_j,
            "median_v": np.array([np.median(v[ok[:, j], j]) for j in range(len(BAND_NAMES))]),
        }
    )
    scale_chart = (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=list(BAND_NAMES), title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("b_j:Q", title="b_j = median |sigma_f| on the training tiles"),
        )
        .properties(height=200)
    )
    deepest = comparison.loc[comparison["b_j"].idxmin()]
    shallowest = comparison.loc[comparison["b_j"].idxmax()]
    mo.vstack(
        [
            mo.md(
                f"""
                ## 5. The asinh view, and why the VAE does not use it

                ```text
                x       = asinh(f / b_j)             b_j = median |sigma_f| per band, training rows
                sigma_x = sigma_f / sqrt(f^2 + b_j^2)
                f       = b_j * sinh(x)              exact inverse
                ```

                Asinh is linear for `|f| < b_j`, logarithmic for `|f| >> b_j`, defined for
                negative flux, and exactly invertible. Checked here over
                {int(ok.sum()):,} measured pixels: the maximum **relative** error of
                `x -> f` is `{rel.max():.1e}` (median `{np.median(rel):.1e}`), i.e. float32
                storage noise.

                `b_j` ranges {deepest["b_j"]:.3f} (`{deepest["band"]}`) to
                {shallowest["b_j"]:.3f} (`{shallowest["band"]}`): a factor
                {shallowest["b_j"] / deepest["b_j"]:.0f} between the shallowest and the
                deepest channel, which is why one global scale would be wrong.

                The VAE does **not** train on `x`; this view is the input of the asinh-PCA
                baseline only (`jpas_da/asinh.py`, `modeling/baselines.py`). A model without
                a noise likelihood needs roughly uniform noise, and asinh provides it; a
                model *with* one does better on the shape/brightness split in section 6.
                """
            ),
            scale_chart,
        ]
    )
    return b_j, sigma_x, x


@app.cell
def _(BAND_INDEX, BAND_NAMES, NORM_BANDS, alt, flux, g, logr, mask, mo, np, pd, sig_logr, v):
    # Recover the normalisation by construction: compare g*r with the stored flux.
    r = np.exp(logr.astype(np.float64))
    roundtrip = np.abs(np.where(mask == 1, g * r[:, None], 0.0) - np.where(mask == 1, flux, 0.0))
    roundtrip = roundtrip[mask == 1] / np.maximum(np.abs(flux[mask == 1]), 1e-12)

    norm_index = [BAND_INDEX[b] for b in NORM_BANDS]
    n_norm = mask[:, norm_index].sum(axis=1)
    # v2 normalises by the RMS of the measured norm bands, so this is exactly 1.
    # It is NOT a unit sphere over all 60 bands: that was v1, and sum_j g_j^2 here
    # ranges from about 6 to 700, which is the v_rms of the norm bands.
    norm_band_g = g.astype(np.float64)[:, norm_index]
    norm_band_mask = mask[:, norm_index]
    norm = np.sqrt((norm_band_g**2 * norm_band_mask).sum(axis=1) / n_norm)
    spread = np.sqrt((g.astype(np.float64) ** 2 * mask).sum(axis=1))
    measured_v = v[mask == 1].astype(np.float64)
    snr = 1.0 / np.maximum(sig_logr, 1e-12)

    mo.md(
        f"""
        ## 6. What the VAE trains on: shape, brightness, noise

        Every band of an object is divided by **one** number, its broad-band brightness:

        ```text
        r_bb     = sqrt( mean over measured {{gSDSS, rSDSS, iSDSS}} of f^2 )
        g_j      = f_j / r_bb                                   the decoder target
        v_j      = sqrt( sigma_j^2 + (0.026 f_j)^2 ) / r_bb     0 where mask_j = 0
        logr     = log r_bb                                     a condition, never predicted
        sig_logr = sqrt( sum_k m_k f_k^2 sigma_k^2 ) / (n r_bb^2)   delta method
        ```

        Why the broad bands and not all 60: over the 54 narrow bands, noise makes up a
        median 69% of the RMS, so a narrow-band brightness would be mostly noise. `logr`
        has median S/N **{np.median(snr):.1f}** (5th percentile {np.percentile(snr, 5):.1f}).

        ### The invariants, checked on these arrays

        | property | measured | expected |
        |---|---|---|
        | `sqrt(mean over measured {{g,r,i}} of g^2)` | {norm.min():.9f} .. {norm.max():.9f} | **1** exactly |
        | `sqrt(sum over all 60 of g^2 * mask)` | {spread.min():.1f} .. {spread.max():.1f} | not 1: v2 is not a unit sphere |
        | `g * r` vs the stored flux | max relative error `{roundtrip.max():.1e}` | float32 storage noise |
        | `v` where masked | {float(np.abs(v[mask == 0]).max()):.1f} | exactly 0 |
        | `v` where measured | p1 {np.percentile(measured_v, 1):.4f}, median {np.median(measured_v):.4f} | > 0 |
        | `logr` | median {np.median(logr):.2f}, p1..p99 {np.percentile(logr, 1):.2f}..{np.percentile(logr, 99):.2f} | spans the sample |

        The one hard invariant is the first row: dividing by the RMS of the measured
        norm bands makes that RMS exactly 1 for every object, whatever its brightness.
        The second row is the warning: unlike the earlier unit-sphere design, the
        60-band norm is *not* 1, so `g` keeps the narrow-versus-broad contrast that the
        v1 normaliser threw away.

        A masked band keeps `g = 0` and `v = 0`, so the loss gives it exactly zero
        weight — see section 8. `0.026` is the survey's relative photometric-calibration
        scatter (~0.028 mag), a constant, not a fitted quantity.
        """
    )
    return norm, roundtrip, snr, spread


@app.cell
def _(BANDS, BAND_NAMES, BLOCK_ORDER, alt, g, logr, mask, mo, np, pd, v):
    rng = np.random.default_rng(0)
    pixels = rng.choice(mask.size, size=min(18_000, mask.size), replace=False)
    scatter_frame = pd.DataFrame(
        {
            "g": g.ravel()[pixels].astype(np.float64),
            "v": v.ravel()[pixels].astype(np.float64),
            "measured": mask.ravel()[pixels] == 1,
        }
    ).query("measured")
    dots = (
        alt.Chart(scatter_frame)
        .mark_circle(size=3, opacity=0.25)
        .encode(
            x=alt.X("g:Q", title="g = f / r_bb (shape)"),
            y=alt.Y("v:Q", title="v = sigma_g (shape error)", scale=alt.Scale(type="log")),
            tooltip=["g", "v"],
        )
        .properties(height=280, width=430, title="the noise model, cell by cell")
    )

    band_frame = pd.DataFrame(
        {"band": list(BAND_NAMES), "wavelength_nm": [b.wavelength_aa / 10.0 for b in BANDS]}
    )
    band_frame["median_g"] = [np.median(g[mask[:, j] == 1, j]) for j in range(len(BANDS))]
    band_frame["lo"] = [np.percentile(g[mask[:, j] == 1, j], 16) for j in range(len(BANDS))]
    band_frame["hi"] = [np.percentile(g[mask[:, j] == 1, j], 84) for j in range(len(BANDS))]
    band_frame["block"] = [b.block for b in BANDS]
    mean_sed = (
        alt.Chart(band_frame)
        .mark_line(point=alt.OverlayMarkDef(size=12))
        .encode(
            x=alt.X("wavelength_nm:Q", title="wavelength [nm]"),
            y=alt.Y("median_g:Q", title="median g"),
            color=alt.Color("block:N", sort=list(BLOCK_ORDER), title="block"),
            tooltip=["band", alt.Tooltip("median_g:Q", format=".3f")],
        )
        .properties(height=280, width=430, title="median shape across the sample")
    )
    band_band = (
        alt.Chart(band_frame)
        .mark_area(opacity=0.2)
        .encode(
            x=alt.X("wavelength_nm:Q"),
            y=alt.Y("lo:Q"),
            y2="hi:Q",
            color=alt.Color("block:N", sort=list(BLOCK_ORDER), legend=None),
        )
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 7. The two channels, side by side

                Left: every measured cell of an 18k random sample, in the plane the
                likelihood lives in. The floor at small `v` is the 0.026 calibration
                term; the cloud to the left is the noise-dominated regime, where `|g|` is
                small and `v` is comparatively large.

                Right: the sample's median SED shape with the 16–84 percentile band. This
                is the "average object" a decoder that ignored its latent would have to
                produce — the baseline any reconstruction number must beat.
                """
            ),
            mo.hstack([dots, mean_sed + band_band], justify="space-around"),
        ]
    )
    return


@app.cell
def _(g, mask, mo, np, object_id, objects):
    n_measured = (mask == 1).sum(axis=1)
    sp_class = objects["spCLASSsdss"].to_numpy()
    candidates = {}
    for cls in ("GALAXY", "STAR", "QSO"):
        members = np.flatnonzero(sp_class == cls)
        if members.size:
            candidates[f"{cls} (median bands measured)"] = int(members[len(members) // 2])
    candidates["median object"] = int(np.argsort(-n_measured)[len(n_measured) // 2])
    candidates["noisiest: fewest bands measured"] = int(np.argmin(n_measured))
    options = {
        f"{object_id[i]} - {label} | {int(n_measured[i])} bands measured": i
        for label, i in candidates.items()
    }
    picker = mo.ui.dropdown(options=options, value=next(iter(options)), label="object")
    return n_measured, picker


@app.cell
def _(
    BANDS,
    BAND_NAMES,
    alt,
    g,
    logr,
    mask,
    mo,
    n_measured,
    np,
    object_id,
    objects,
    pd,
    picker,
    v,
):
    chosen = picker.value
    in_g = g[chosen].astype(np.float64)
    in_v = v[chosen].astype(np.float64)
    chosen_measured = mask[chosen] == 1
    r_object = float(np.exp(logr[chosen]))

    frame = pd.DataFrame(
        {
            "band": list(BAND_NAMES),
            "wavelength_nm": [b.wavelength_aa / 10.0 for b in BANDS],
            "g": in_g,
            "v": np.where(chosen_measured, in_v, np.nan),
            "flux": in_g * r_object,
            "measured": chosen_measured,
        }
    )

    def _chart(y: str, title: str) -> alt.LayerChart:
        base = alt.Chart(frame).encode(
            x=alt.X("wavelength_nm:Q", title="wavelength [nm]"),
            tooltip=["band", "measured", alt.Tooltip(f"{y}:Q", format=".4g")],
        )
        points = base.mark_point(size=18).encode(
            y=alt.Y(f"{y}:Q", title=title),
            color=alt.Color(
                "measured:N",
                title="mask",
                scale=alt.Scale(domain=[True, False], range=["#4c78a8", "#e45756"]),
            ),
        )
        bars = (
            base.transform_filter(alt.datum.measured)
            .mark_rule(opacity=0.45)
            .encode(y=f"{y}:Q", y2="v:Q")
        )
        return (points + bars).properties(height=200, width=620, title=title)

    spectral = objects["spCLASSsdss"][chosen]
    photo_z = objects["zph"][chosen]
    spec_z = objects["zsp"][chosen]
    class_text = f"`{spectral}`" if pd.notna(spectral) else "no spectroscopic class"
    z_text = (
        ", ".join(
            part
            for part in (
                f"`zsp = {spec_z:.3f}`" if pd.notna(spec_z) else "",
                f"`zph = {photo_z:.3f}`" if pd.notna(photo_z) else "",
            )
            if part
        )
        or "no redshift"
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 8. One object: exactly what the network sees

                Every dot is one of the 60 channels; the bars are `±v`, the scale the
                likelihood uses. Red marks a masked band: present in the tensor as
                `g = 0, v = 0`, contributing nothing to the loss.

                The top panel is the **decoder target**. The bottom panel multiplies it back
                by the object's own `r_bb = exp(logr)`, which is the only thing that makes
                an SED look like a spectrum instead of a shape.
                """
            ),
            picker,
            _chart("g", f"{object_id[chosen]}: g (shape) with ±v"),
            _chart("flux", "the same object in archive flux units (g * r_bb)"),
            mo.md(
                f"""
                `{object_id[chosen]}` is {class_text} with {z_text} — labels for
                interpretation only, never inputs. It has
                **{int(chosen_measured.sum())}/{len(BAND_NAMES)} measured bands**,
                `logr = {float(logr[chosen]):.2f}` (`r_bb = {r_object:.2f}` in archive flux
                units) and **{int((~chosen_measured).sum())} masked** channel(s).
                """
            ),
        ]
    )
    return


@app.cell
def _(BANDS, BAND_NAMES, alt, g, logr, mask, mo, np, object_id, pd):
    rng_grid = np.random.default_rng(7)
    fully_measured = np.flatnonzero((mask == 1).all(axis=1))
    sample = rng_grid.choice(fully_measured, size=min(12, len(fully_measured)), replace=False)
    long = pd.DataFrame(
        {
            "object_id": np.repeat([object_id[i] for i in sample], len(BAND_NAMES)),
            "band": list(BAND_NAMES) * len(sample),
            "wavelength_nm": np.tile([b.wavelength_aa / 10.0 for b in BANDS], len(sample)),
            "g": g[sample].ravel().astype(np.float64),
        }
    )
    grid = (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=8))
        .encode(
            x=alt.X("wavelength_nm:Q", title="wavelength [nm]"),
            y=alt.Y("g:Q", title="g"),
            tooltip=["object_id", "band", alt.Tooltip("g:Q", format=".3f")],
            facet=alt.Facet("object_id:N", columns=4, title=None),
        )
        .properties(width=150, height=85)
    )

    many = rng_grid.choice(fully_measured, size=min(300, len(fully_measured)), replace=False)
    spaghetti_frame = pd.DataFrame(
        {
            "object_id": np.repeat([object_id[i] for i in many], len(BAND_NAMES)),
            "wavelength_nm": np.tile([b.wavelength_aa / 10.0 for b in BANDS], len(many)),
            "g": g[many].ravel().astype(np.float64),
            "logr": np.repeat(logr[many], len(BAND_NAMES)).astype(np.float64),
        }
    )
    spaghetti = (
        alt.Chart(spaghetti_frame)
        .mark_line(size=0.7, opacity=0.18)
        .encode(
            x=alt.X("wavelength_nm:Q", title="wavelength [nm]"),
            y=alt.Y("g:Q", title="g"),
            detail="object_id:N",
            color=alt.Color("logr:Q", title="log r_bb", scale=alt.Scale(scheme="viridis")),
        )
        .properties(height=300, width=620, title="300 random objects, coloured by brightness")
    )
    measured_g = np.where(mask == 1, g, np.nan).astype(np.float64)
    level = np.nanmean(measured_g, axis=1, keepdims=True)
    deviation = measured_g - level  # per-band colour, level removed
    total_var = np.nanvar(measured_g)
    mo.vstack(
        [
            mo.md(
                f"""
                ## 9. Many objects: is the shape channel actually varied?

                Twelve complete spectra, then 300 objects overlaid: the vertical spread at
                each wavelength is the SED diversity the model has to encode, and the
                colour is brightness.

                Read the colour as a control: if brightness leaked into `g`, the curves
                would be ordered by colour. They are not, because `g` is normalised by the
                object's own `r_bb` — two objects of the same class at different distances
                land on nearly the same curve. That is why `logr` is a separate condition
                rather than a 61st channel.

                | component | share of variance in `g` |
                |---|---|
                | per-object level (mean of `g`) | **{100 * np.nanvar(level) / total_var:.0f}%** |
                | per-band shape (colour) | **{100 * np.nanvar(deviation) / total_var:.0f}%** |

                A large fraction of `g`'s variance is one number per object. That is why a
                small latent already reconstructs a lot, and why every reconstruction number
                must be compared with the mean-shape baseline (section 11).
                """
            ),
            grid,
            spaghetti,
        ]
    )
    return (fully_measured,)


@app.cell
def _(BAND_NAMES, alt, mask, mo, np, pd):
    per_band = pd.DataFrame({"band": list(BAND_NAMES), "masked_fraction": 1.0 - mask.mean(axis=0)})
    chart = (
        alt.Chart(per_band)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=list(BAND_NAMES), title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y(
                "masked_fraction:Q",
                title="fraction of objects masked",
                axis=alt.Axis(format="%"),
            ),
            tooltip=["band", alt.Tooltip("masked_fraction:Q", format=".4f")],
        )
        .properties(height=200)
    )
    worst = per_band.loc[per_band["masked_fraction"].idxmax()]
    mo.vstack(
        [
            mo.md(
                f"""
                ## 10. The mask channel

                `mask` is 1 where the archive quoted a non-zero relative error. It is an
                input channel *and* a loss selector, so the model can tell "unmeasured band"
                from "measured, flux near zero" — without it those two look identical after
                `g = 0`.

                * {int((mask == 0).any(axis=1).sum()):,} objects have at least one masked channel
                * {int((mask == 0).all(axis=1).sum())} objects are fully masked
                * {int((mask == 0).sum()):,} pixels ({100 * (mask == 0).mean():.2f}%)
                * worst band: `{worst["band"]}` at {100 * worst["masked_fraction"]:.2f}%

                The loss multiplies by `mask`, so a masked cell has zero weight whatever the
                decoder predicts. Nothing imputes these pixels.
                """
            ),
            chart,
        ]
    )
    return


@app.cell
def _(arrays, mo, model_data, np):
    rows = []
    for k in range(len(np.unique(arrays["tile_id"]))):
        rotation_parts = model_data.split_indices(arrays, k)
        tiles = sorted(int(t) for t in np.unique(arrays["tile_id"][rotation_parts["test"]]))
        tile_text = ", ".join(map(str, tiles))
        rows.append(
            f"| {k} | {tile_text} | {len(rotation_parts['train']):,} | "
            f"{len(rotation_parts['val']):,} | {len(rotation_parts['test']):,} | "
            f"{len(rotation_parts['holdout']):,} |"
        )
    table = (chr(10) + " " * 8).join(rows)
    mo.md(
        f"""
        ## 11. Splits: rotate over the pointings

        | rotation | test tile | train | val | test | holdout |
        |---|---|---|---|---|---|
        {table}

        The unit of independence is the pointing, not the object. Objects are never shared
        between the four miniJPAS pointings, but neighbours inside one pointing share sky
        background, crowding and a common photometric calibration, so a random row split
        would train and score on the same noise. `holdout` and the `severe` quality class
        get no role at all, in any rotation.

        The pointings differ in depth, so a rotation's score is partly a depth-mismatch
        score. That is the intended reading: it is the production situation, applying a
        model to a new field.

        The roles are recomputed in code (`jpas_da.splits.assign_roles`) from the rotation
        number, so `splits.parquet` is a convenience for joins and cannot drift from what
        training uses.
        """
    )
    return


@app.cell
def _(g, mask, mo, np, sig_logr, v):
    eps = 0.1
    w2 = v.astype(np.float64) ** 2 + eps**2 + (g.astype(np.float64) * sig_logr[:, None]) ** 2
    w2 = w2 * mask
    safe = np.where(mask == 1, w2, np.nan)
    fisher = np.nansum(1.0 / safe, axis=1)
    fisher_order = np.argsort(-fisher)
    fisher_total = fisher.sum()
    top5 = fisher_order[: max(1, int(0.05 * len(fisher_order)))]
    mo.md(
        f"""
        ## 12. The noise floor, and why `v` alone is not enough

        The archive's error is **purely relative**, so `v` shrinks with the flux and a
        near-zero-flux band of a very bright object would claim enormous precision. The
        likelihood therefore adds a fixed floor relative to the object's brightness:

        ```text
        w^2 = v^2 + eps^2 + (g * sig_logr)^2       eps = 0.1 (configs/vae.yaml), 0 where mask = 0
        ```

        | term | what it is |
        |---|---|
        | `v^2` | the propagated measurement error, already carrying the 0.026 calibration term |
        | `eps^2` | no band is known better than `eps * r_bb`; fixed, not learned |
        | `(g*sig_logr)^2` | the error that dividing by a noisy `r_bb` adds to every band (`loss.common_mode`) |

        Measured on these arrays: the **top 5% of objects hold
        {100 * fisher[top5].sum() / fisher_total:.0f}%** of the total Fisher weight
        `sum_j 1/w_j^2`, and the brightest 1% hold
        {100 * fisher[fisher_order[: max(1, int(0.01 * len(fisher_order)))]].sum() / fisher_total:.0f}%. That
        concentration is what `eps` exists to limit; it is a knob
        (`SET="--set loss.eps=0.03"`), and `references/pipeline-v2-plan.md` § B3 records
        how 0.1 was chosen.

        `eps` is **not** stored in `vae_inputs.npz`: a stored floor would have been fitted
        on every object including the held-out tile. It is written into `models/vae.pt`
        so `predict.py` uses the same one as training.
        """
    )
    return eps, w2


@app.cell
def _(MODELS_DIR, REPORTS_DIR, build_model, json, mo, model_data, np, torch):
    model_path = MODELS_DIR / "vae.pt"
    metrics_path = REPORTS_DIR / "metrics.json"
    if not model_path.exists():
        diagnostics = None
        mo.md(f"`{model_path}` not found — run `make train` or `make train-debug` first.")
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        model = build_model(checkpoint["model"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        arrays_ckpt = model_data.load(checkpoint["config"]["data"]["path"])
        channels = tuple(checkpoint["channels"])
        conditions = tuple(checkpoint["conditions"])
        blocks = model_data.model_inputs(arrays_ckpt, channels, conditions)
        rotation = int(checkpoint["rotation"])
        parts = model_data.split_indices(arrays_ckpt, rotation)

        with torch.no_grad():
            out = model(
                torch.from_numpy(blocks["narrow"]),
                torch.from_numpy(blocks["other"]),
                torch.from_numpy(blocks["cond"]),
                sample=False,
            )
        g_hat = out["g_hat"].numpy()
        g_arr = arrays_ckpt["g"]
        mask_arr = arrays_ckpt["mask"]
        test_rows = parts["test"]

        def _rmse(pred, idx):
            resid = (pred[idx] - g_arr[idx]) ** 2 * mask_arr[idx]
            return float(np.sqrt(resid.sum(axis=1) / mask_arr[idx].sum(axis=1)).mean())

        train_mask = mask_arr[parts["train"]]
        train_mean = (g_arr[parts["train"]] * train_mask).sum(axis=0) / train_mask.sum(axis=0)
        train_mean = train_mean / np.linalg.norm(train_mean)
        mu_z = out["mu_z"].numpy()
        diagnostics = {
            "debug": bool(metrics.get("debug")),
            "epoch": int(checkpoint["epoch"]),
            "rotation": rotation,
            "channels": channels,
            "conditions": conditions,
            "latent_dim": int(checkpoint["model"]["latent_dim"]),
            "eps": checkpoint["eps"],
            "train_rmse": _rmse(g_hat, parts["train"]),
            "test_rmse": _rmse(g_hat, test_rows),
            "mean_rmse": _rmse(np.broadcast_to(train_mean, g_arr.shape), test_rows),
            "latent_std": mu_z.std(axis=0),
            "latent_logr_corr": [
                float(np.corrcoef(mu_z[:, k], arrays_ckpt["logr"])[0, 1])
                for k in range(mu_z.shape[1])
            ],
        }
    return (diagnostics,)


@app.cell
def _(diagnostics, mo):
    if diagnostics is None:
        output = mo.md("")
    else:
        kind = "**debug** run (`make train-debug`)" if diagnostics["debug"] else "full run"
        corr = ", ".join(
            f"`z_{k}` {c:+.2f}" for k, c in enumerate(diagnostics["latent_logr_corr"])
        )
        beats = diagnostics["test_rmse"] < diagnostics["mean_rmse"]
        verdict = (
            "better than the mean-shape baseline"
            if beats
            else "**worse than the mean-shape baseline**"
        )
        note = (
            "Read this as a pipeline check, not a result: two epochs cannot beat a "
            "one-parameter baseline, and a nearly flat latent spread means the posterior "
            "has barely moved off its prior. Run `make train` for a real number."
            if diagnostics["debug"]
            else "A full run: this is where the model's quality can be judged."
        )
        output = mo.md(
            f"""
            ## 13. Checking the checkpoint against the data it trained on

            `models/vae.pt` is a {kind}: epoch {diagnostics["epoch"]},
            rotation {diagnostics["rotation"]}, channels `{diagnostics["channels"]}`,
            conditions `{diagnostics["conditions"]}`, latent width
            {diagnostics["latent_dim"]}, `eps = {diagnostics["eps"]}`.

            | | masked RMSE of `g` on the **test tile** |
            |---|---|
            | decoder, training tiles | {diagnostics["train_rmse"]:.4f} |
            | decoder, test tile | {diagnostics["test_rmse"]:.4f} |
            | "predict the mean shape", test tile | {diagnostics["mean_rmse"]:.4f} |

            The decoder is {verdict}.

            Latent spread per dimension:
            {", ".join(f"{s:.3f}" for s in diagnostics["latent_std"])}.
            Correlation of each latent with brightness: {corr}.

            {note}
            """
        )
    mo.output.replace(output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What must not go in, and what the pipeline guarantees

    | do not feed | why |
    |---|---|
    | `MAGrSDSS` | exactly `26.1147 - 2.5*log10(rSDSS)`; a duplicate of a channel |
    | `angDist` | non-null exactly on the SDSS-cross-matched rows: encodes label availability |
    | `zsp`, `zph`, `avg_zph`, `e_*` | labels the model is meant to help predict |
    | `spCLASSsdss`, `photoCLASSsdss` | same |
    | `ALPHA_J2000`, `DELTA_J2000`, `flag_rSDSS` | position and selection, not photometry |

    All of them stay in `objects.parquet` so downstream inference can join them back on
    `object_id`.

    | requirement | enforced by |
    |---|---|
    | no NaN in any model channel | `features.normalise` raises on non-finite `g`/`v`/`logr` |
    | `v == 0` exactly where `mask == 0` | constructed that way, asserted in `tests/test_data.py` |
    | canonical band order everywhere | one definition, `jpas_da.bands.BAND_NAMES` |
    | the four rotations partition the pool | `tests/test_data.py`, `assign_roles` |
    | holdout never appears in a role | `assign_roles`, asserted in `tests/test_data.py` |
    | the checkpoint carries its own architecture | `build_model` takes every value from it |
    | `predict.py` reuses the training `eps` | `eps` is written into the checkpoint |

    ### Reading the rest of the project

    * `references/getting-started.md` — install, data, first run.
    * `references/data_dictionary.md` — every column and count.
    * `references/pipeline-v2-plan.md` — the decisions and the measurements behind them.
    * `jpas_da/features.py` — the normalisation, in ~50 lines.
    * `jpas_da/modeling/train.py` — the loop, the checkpoint, W&B.
    * `models/vae_latents.parquet` — what `make predict` writes; start any downstream
      analysis from there rather than re-running the network.
    """)
    return


if __name__ == "__main__":
    app.run()
