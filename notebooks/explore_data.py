import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from jpas_da import features
    from jpas_da.bands import BANDS, BAND_NAMES, BLOCK_ORDER, N_BANDS
    from jpas_da.config import (
        INTERIM_DATA_DIR,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        REPORTS_DIR,
    )
    from jpas_da.modeling import data as model_data

    return (
        BANDS,
        BAND_NAMES,
        BLOCK_ORDER,
        INTERIM_DATA_DIR,
        MODELS_DIR,
        N_BANDS,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        REPORTS_DIR,
        alt,
        features,
        model_data,
        mo,
        np,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The data: from the SQL export to one training row

    This notebook follows a single object end to end and then shows what a batch
    of them looks like, so that every number the VAE consumes can be traced back
    to a column in the export.

    Two representations exist, and only one of them trains:

    | artifact | built by | holds | used by |
    |---|---|---|---|
    | `data/raw/miniJpas_fluxes_errors_redshifts_full.csv` | the archive | 64,327 rows, 60 fluxes + 60 relative errors + metadata | `dataset.py` only |
    | `data/interim/objects.parquet` | `make interim` | one row per object: flux, **absolute** sigma, validity mask, metadata | `features`, `splits`, `modeling.data` |
    | `data/interim/band_scales.parquet` | `make interim` | per-band soft scale `b_j` | `features` |
    | `data/interim/folds.parquet` | `make splits` | leave-one-tile-out fold per object | `modeling.data`, `train`, `predict` |
    | `data/processed/jpas_asinh.parquet` | `make features` | **exploratory** asinh view: `x`, `sigma_x`, `mask` | this notebook, tests |
    | `data/processed/vae_inputs.npz` | `make model-data` | `g`, `v`, `mask`, `logr`, `sig_logr`, `fold` | **`train.py`, `predict.py`** |
    | `models/vae.pt`, `reports/metrics.json` | `make train` | weights, architecture, noise floor, history | `predict.py` |
    | `data/processed/vae_latents.parquet` | `make predict` | `z_0..z_5`, `l_mu`, reconstruction loss | downstream analysis |

    The important structural fact: **training never reads `jpas_asinh.parquet`.**
    It reads `vae_inputs.npz` (`jpas_da/modeling/data.py`). Sections 1-5 describe
    the asinh view because it is how the flux is understood and checked; sections
    6-10 describe the arrays that are actually optimised.

    Labels (`zsp`, `zph`, `spCLASSsdss`, `photoCLASSsdss`) and position/magnitude
    columns are carried in `objects.parquet` for inference and diagnostics and are
    never fed to the model.
    """)
    return


@app.cell
def _(INTERIM_DATA_DIR, features, model_data, pd):
    # The exploratory representation (asinh) and the training arrays (RMS).
    arrays = features.load_arrays()
    model_arrays = model_data.load()

    objects = pd.read_parquet(INTERIM_DATA_DIR / "objects.parquet")
    scales = pd.read_parquet(INTERIM_DATA_DIR / "band_scales.parquet")
    folds = pd.read_parquet(INTERIM_DATA_DIR / "folds.parquet")
    return arrays, folds, model_arrays, objects, scales


@app.cell
def _(arrays, model_arrays, np, objects, scales):
    # Every array below is in canonical band order and aligned row by row: the
    # same object sits at the same index in all of them.
    assert (objects["object_id"].to_numpy() == arrays["object_id"]).all()
    assert (model_arrays["object_id"] == arrays["object_id"]).all()

    x = arrays["x"]  # asinh(f / b_j) * 2
    sigma_x = arrays["sigma_x"]  # propagated uncertainty, NaN where masked
    mask = arrays["mask"]  # 1 = uncertainty measured
    fold = arrays["fold"]
    tile_id = arrays["tile_id"]
    object_id = arrays["object_id"]

    g = model_arrays["g"]  # f / r        -- the decoder target
    v = model_arrays["v"]  # sigma_g      -- the likelihood's scale
    logr = model_arrays["logr"]  # log r    -- brightness
    sig_logr = model_arrays["sig_logr"]  # its statistical error

    b_j_values = scales["b_j"].to_numpy(dtype=np.float64)
    band_index = np.arange(len(b_j_values))
    return (
        b_j_values,
        band_index,
        fold,
        g,
        logr,
        mask,
        object_id,
        sigma_x,
        sig_logr,
        tile_id,
        v,
        x,
    )


@app.cell
def _(N_BANDS, fold, mask, mo, np, object_id, tile_id, x):
    n_objects = x.shape[0]
    n_masked = int((mask == 0).sum())
    n_complete = int((mask == 1).all(axis=1).sum())
    tiles = np.unique(tile_id)
    mo.md(
        f"""
        ## 1. The sample at a glance

        | | |
        |---|---|
        | objects | **{n_objects:,}** |
        | channels per object | **{N_BANDS}** (54 narrow + `uJAVA` + 4 broad + `J1007`) |
        | tiles / folds | **{len(tiles)}** pointings, one leave-one-out fold each: {", ".join(str(t) for t in tiles)} |
        | objects with all {N_BANDS} errors measured | {n_complete:,} ({100 * n_complete / n_objects:.2f}%) |
        | objects with at least one masked channel | {n_objects - n_complete:,} |
        | masked pixels | {n_masked:,} ({100 * n_masked / mask.size:.2f}%) |
        | object id | `<tile_id>-<number>`, e.g. `{object_id[0]}` |
        | folds present | {len(np.unique(fold))} |
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How to load it

    ```python
    from jpas_da.modeling.data import load      # what training uses

    a = load()                                  # data/processed/vae_inputs.npz
    g, v, mask = a["g"], a["v"], a["mask"]      # (n_objects, 60) float32 / int8
    logr, sig_logr = a["logr"], a["sig_logr"]   # (n_objects,)     float32
    fold = a["fold"]                            # leave-one-tile-out assignment
    ```

    ```python
    from jpas_da.features import load_arrays    # the asinh view

    b = load_arrays()                           # data/processed/jpas_asinh.parquet
    x, sigma_x, mask = b["x"], b["sigma_x"], b["mask"]
    ```

    Band order is the single definition in `jpas_da.bands.BAND_NAMES`, identical
    in every array, so no column matching can drift. `load()` refuses a file that
    is missing any of `g, v, mask, logr, sig_logr, fold, object_id`.
    """)
    return


@app.cell
def _(BANDS, BLOCK_ORDER, alt, mo, pd):
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
        .properties(height=300, title="channel index -> wavelength: the 54 narrow bands are the spectrum")
    )
    counts = band_table["block"].value_counts().reindex(list(BLOCK_ORDER))
    span = band_table["wavelength_nm"].max() - band_table["wavelength_nm"].min()
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

                {span:.0f} nm of contiguous coverage in {len(band_table)} channels.
                Adjacent narrow bands overlap, which is why the encoder is a 1-D
                convolution along this axis rather than a dense layer.
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
def _(BAND_NAMES, RAW_DATA_DIR, np, objects, pd):
    # One deliberate read of the 75 MB export: every cleaning claim in the next
    # section is checked here rather than quoted.
    raw = pd.read_csv(
        RAW_DATA_DIR / "miniJpas_fluxes_errors_redshifts_full.csv",
        skiprows=1,
        usecols=[
            "tile_id",
            "number",
            *BAND_NAMES,
            *[f"e{b}" for b in BAND_NAMES],
            "zsp",
            "zph",
            "MAGrSDSS",
        ],
    )
    relerr = raw[[f"e{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float64)
    audit = {
        "rows_raw": len(raw),
        "rows_objects": len(objects),
        "duplicates": int(raw.duplicated(["tile_id", "number"]).sum()),
        "relerr_zero": float((relerr == 0).mean()),
        "zsp_negative": int((raw["zsp"] < 0).sum()),
        "zph_sentinel": int((raw["zph"] < 0).sum()),
        "mag_sentinel": int((raw["MAGrSDSS"] > 30).sum()),
        "zsp_positive": int((raw["zsp"] > 0).sum()),
    }
    return (audit,)


@app.cell
def _(BAND_NAMES, audit, mask, mo, np, objects, pd):
    per_tile = pd.DataFrame(
        {"tile_id": objects["tile_id"].to_numpy(), "masked": (mask == 0).any(axis=1)}
    ).groupby("tile_id")["masked"].mean()
    # mo.md de-indents with textwrap.dedent, so every interpolated newline has to
    # carry the block's indent; a column-0 line turns the whole block into code.
    newline_indent = chr(10) + " " * 16
    tile_rows = newline_indent.join(
        f"| {int(tile)} | {100 * share:.2f}% |" for tile, share in per_tile.items()
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. What `dataset.py` has to repair before anything else works

                The export was written by an SQL query (its first line is the query
                itself) and is not analysis-ready. Measured on the shipped file:

                | defect | count | what the pipeline does |
                |---|---|---|
                | duplicated `(tile_id, number)` | {audit["duplicates"]} | keep the smallest `angDist` row; fluxes are identical within a group |
                | `e<band> == 0` (uncertainty not computed) | {100 * audit["relerr_zero"]:.2f}% of pixels | `sigma = NaN`, `mask = 0` |
                | `zsp < 0` / `zph < 0` sentinels | {audit["zsp_negative"]} / {audit["zph_sentinel"]} | `NaN`; `zsp == 0` is kept only for STARs |
                | `MAGrSDSS > 30` sentinel | {audit["mag_sentinel"]} | `NaN` |

                Rows: **{audit["rows_raw"]:,} -> {audit["rows_objects"]:,}** objects.

                The `e<band> == 0` case is the dangerous one. Zero relative error
                reads as *infinitely precise*, but it means *not computed*: those
                pixels sit on faint sources in the noisiest bands, and unevenly
                across pointings:

                | tile | objects with >=1 masked channel |
                |---|---|
                {tile_rows}

                `build_photometry` therefore produces `sigma` and `mask` together:
                `sigma` is `NaN` exactly where `mask == 0`, so the two can never
                disagree. An `f/sigma` representation would silently turn these
                pixels into enormous signal-to-noise; asinh plus an explicit mask
                cannot.
                """
            ),
            mo.md(
                """
                `sigma_f = |flux * FLUX_RELERR_AUTO|`, because the export's error
                column is a **signed relative** error, not an absolute one. The flux
                unit is not uJy: the zeropoint is 26.1147, and `MAGrSDSS` is exactly
                `26.1147 - 2.5*log10(rSDSS)`, which is why it is kept out of every
                feature array.
                """
            ),
        ]
    )
    return


@app.cell
def _(alt, b_j_values, mask, mo, np, pd, x):
    # Recover raw flux exactly: the stored x is asinh(f / b_j) * 2.
    flux = b_j_values * np.sinh(x.astype(np.float64) / 2.0)
    pixel_ok = mask == 1
    flux_measured = flux[pixel_ok]

    def _hist(values: np.ndarray, bins: int = 90) -> pd.DataFrame:
        counts, edges = np.histogram(values, bins=bins)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

    hist_x = (
        alt.Chart(_hist(x.astype(np.float64)[pixel_ok]))
        .mark_bar()
        .encode(
            x=alt.X("left:Q", title="x = asinh(f/b) * 2"),
            x2="right:Q",
            y=alt.Y("count:Q", title="pixels", scale=alt.Scale(type="log")),
        )
        .properties(height=220, width=330, title="model coordinate")
    )
    hist_flux = (
        alt.Chart(_hist(np.sign(flux_measured) * np.log10(np.abs(flux_measured) + 1.0)))
        .mark_bar()
        .encode(
            x=alt.X("left:Q", title="sign(f) * log10(|f| + 1)  [raw flux units]"),
            x2="right:Q",
            y=alt.Y("count:Q", title="pixels", scale=alt.Scale(type="log")),
        )
        .properties(height=220, width=330, title="same pixels, physical flux")
    )
    negative = float((flux_measured < 0).mean())
    mo.vstack(
        [
            mo.md(
                f"""
                ## 4. Why the flux is transformed at all

                * **{100 * negative:.1f}% of measured pixels have negative flux** -
                  noise, not astrophysics, but real data. A logarithm is undefined
                  there, so a log representation would need an arbitrary floor or an
                  imputation.
                * The bright end spans four decades. On raw flux, `sum(f^2)` is
                  dominated by a handful of pixels, so any Euclidean loss would fit
                  those and ignore the rest.

                `asinh` is linear while `|f| < b_j` (keeps signs, no clipping) and
                logarithmic for `|f| >> b_j`, and it is exactly invertible. The two
                histograms are the same pixels before and after.
                """
            ),
            mo.hstack([hist_x, hist_flux], justify="space-around"),
        ]
    )
    return


@app.cell
def _(BAND_NAMES, alt, b_j_values, mo, scales):
    scale_chart = (
        alt.Chart(scales)
        .mark_bar()
        .encode(
            x=alt.X(
                "band:N", sort=list(BAND_NAMES), title=None, axis=alt.Axis(labelAngle=-90)
            ),
            y=alt.Y("b_j:Q", title="b_j = median |sigma_f| (flux units)"),
            tooltip=["band", alt.Tooltip("b_j:Q", format=".3f"), "n_sigma_used"],
        )
        .properties(height=240)
    )
    band_b = scales.set_index("band")["b_j"]
    deepest, shallowest = band_b.idxmin(), band_b.idxmax()
    mo.vstack(
        [
            mo.md(
                f"""
                ## 5. The per-band soft scale `b_j`

                `b_j` is the **median absolute flux uncertainty of that band over
                the whole sample** - the flux at which the band stops being
                noise-dominated. It is the pivot of the asinh transform, so each
                channel is compressed on its own scale instead of one global number
                trying to fit bands that differ by orders of magnitude in depth.

                | band | `b_j` |
                |---|---|
                | `{deepest}` (deepest) | {band_b[deepest]:.3f} |
                | `{shallowest}` (shallowest) | {band_b[shallowest]:.3f} |

                Ratio **{band_b.max() / band_b.min():.0f}x**.

                It is global, not per tile, on purpose: the pointings differ in
                depth, and leave-one-tile-out is meant to expose that, not to have it
                normalised away. This is a *pivot*, not a fitted model parameter, so
                using all objects for it is not leakage - it carries no label and no
                per-object information.
                """
            ),
            scale_chart,
        ]
    )
    return


@app.cell
def _(BAND_NAMES, INTERIM_DATA_DIR, b_j_values, features, mask, mo, np, pd, x):
    truth = pd.read_parquet(
        INTERIM_DATA_DIR / "objects.parquet", columns=[f"flux_{b}" for b in BAND_NAMES]
    ).to_numpy(dtype=np.float64)
    recovered = features.asinh_inverse(x / features.PRESENTATION_SCALE, b_j_values)
    inverse_ok = mask == 1
    residual = np.abs(recovered - truth)[inverse_ok] / np.maximum(np.abs(truth[inverse_ok]), 1e-12)
    mo.md(
        f"""
        ### The transform, and its exact inverse

        ```python
        x       = asinh(f / b_j) * 2                     # exploratory view
        sigma_x = sigma_f / sqrt(f^2 + b_j^2) * 2        # chain rule: d asinh(f/b)/df = 1/sqrt(f^2+b^2)
        f       = b_j * sinh(x / 2)                      # exact inverse
        ```

        Checked here, live: over the {int(inverse_ok.sum()):,} measured pixels the
        maximum **relative** error of `x -> f` is `{residual.max():.1e}` (median
        `{np.median(residual):.1e}`), i.e. float32 storage noise. `features.build`
        asserts the same thing on every rebuild.

        Masked pixels carry `x = 0` and `sigma_x = NaN` and are excluded above:
        their `x` is 0 *by construction*, so inverting them would invent a flux.
        That is the reason the loss must be masked rather than the reconstruction
        trusted everywhere.

        The factor `2` (`features.PRESENTATION_SCALE`) is monotonic, so it changes
        no distance ordering; it exists so the channels have a convenient numerical
        size and so the file stays invertible.
        """
    )
    return


@app.cell
def _(BAND_NAMES, INTERIM_DATA_DIR, alt, mask, mo, np, pd, sigma_x):
    sigma_f = pd.read_parquet(
        INTERIM_DATA_DIR / "objects.parquet", columns=[f"sigma_{b}" for b in BAND_NAMES]
    ).to_numpy(dtype=np.float64)
    flux_f = pd.read_parquet(
        INTERIM_DATA_DIR / "objects.parquet", columns=[f"flux_{b}" for b in BAND_NAMES]
    ).to_numpy(dtype=np.float64)
    decile = pd.qcut(np.abs(flux_f).ravel(), 10, labels=False, duplicates="drop")
    sigma_x_measured = np.where(mask == 1, sigma_x.astype(np.float64), np.nan)
    in_flux = pd.Series(sigma_f.ravel()).groupby(decile).median()
    in_asinh = pd.Series(sigma_x_measured.ravel()).groupby(decile).median()
    comparison = pd.DataFrame(
        {
            "decile": list(in_flux.index) * 2,
            "median_sigma": list(in_flux.to_numpy()) + list(in_asinh.to_numpy()),
            "space": ["flux units"] * len(in_flux) + ["asinh units"] * len(in_asinh),
        }
    )
    flatness = (
        alt.Chart(comparison)
        .mark_line(point=True)
        .encode(
            x=alt.X("decile:Q", title="decile of |flux| (0 = faintest)", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("median_sigma:Q", title="median sigma", scale=alt.Scale(type="log")),
            color=alt.Color("space:N", title=None),
            tooltip=["space", "decile", alt.Tooltip("median_sigma:Q", format=".3g")],
        )
        .properties(height=260, width=560)
    )
    span_flux = float(in_flux.max() / in_flux.min())
    span_x = float(in_asinh.iloc[:-1].max() / in_asinh.iloc[:-1].min())
    mo.vstack(
        [
            mo.md(
                f"""
                ## 6. The uncertainty survives the transform

                The error is propagated, not discarded: `sigma_x` is the flux error
                mapped through the same function as the flux.

                * in **flux units** the median error grows **{span_flux:.1f}x** from
                  the faintest to the brightest decile;
                * in **asinh units** it varies only **{span_x:.1f}x** over the nine
                  faintest deciles, and drops on the brightest, where the data
                  really are better.

                That near-flatness is what a heteroscedastic Gaussian likelihood
                wants: the noise model does not have to learn "trust bright pixels
                more", the `sigma` channel already says it.
                """
            ),
            flatness,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. What the model actually optimises

    The asinh view is for looking and checking. Training reads
    `data/processed/vae_inputs.npz`, built by `jpas_da/modeling/data.py`, which
    rewrites flux and error as a **per-object brightness** and a **unit-norm
    shape**:

    ```text
    r        = sqrt( sum_j f_j^2 * m_j )                  brightness (masked bands excluded)
    g_j      = f_j / r                                    SED shape, unit norm over m_j = 1
    v_j      = sqrt( sigma_j^2 + (0.026 f_j)^2 ) / r      shape error, 0 where m_j = 0
    logr     = log r                                      brightness target
    sig_logr = sqrt( sum_j m_j g_j^2 v_j^2 )              its statistical error
    ```

    Why this and not `x`:

    * The shape and the brightness are separate physical questions. A model of
      "what kind of object is this" should see `g`, not `g * r`; brightness is a
      distance/luminosity axis and gets its own scalar target.
    * Dividing flux and error by the same per-object scalar leaves every Gaussian
      likelihood term invariant, so this is conditioning, not modelling. It uses
      the object's own numbers only - no sample statistic - so it cannot leak
      across folds.
    * `0.026` is the survey's relative photometric-calibration scatter (~0.028
      mag), a constant, not a fitted quantity.
    * A masked band gets `g = 0, v = 0`, so it contributes exactly zero to the
      loss instead of an invented uncertainty.

    Note the one thing **not** stored here: the additive per-band error floor.
    It must be fitted on the training fold only, so `train.py` derives it at run
    time (section 11).
    """)
    return


@app.cell
def _(g, logr, mask, mo, np, sig_logr, v):
    norm = np.sqrt((g.astype(np.float64) ** 2 * mask).sum(axis=1))
    r = np.exp(logr.astype(np.float64))
    measured_v = v[mask == 1].astype(np.float64)
    mo.md(
        f"""
        ### The invariants, checked

        | property | measured | expected |
        |---|---|---|
        | `sqrt(sum(g^2 * mask))` | {norm.min():.6f} .. {norm.max():.6f} | 1.0 |
        | `v` where masked | {float(np.abs(v[mask == 0]).max()):.1f} | 0 |
        | `v` where measured | p1 {np.percentile(measured_v, 1):.4f}, median {np.median(measured_v):.4f}, p99 {np.percentile(measured_v, 99):.4f} | > 0 |
        | `logr` | p1 {np.percentile(logr, 1):.2f}, median {np.median(logr):.2f}, p99 {np.percentile(logr, 99):.2f} | log of a positive norm |
        | `sig_logr` | median {np.median(sig_logr):.3f}, p99 {np.percentile(sig_logr, 99):.3f} | > 0 |

        `r` spans {r.max() / r.min():.0f}x across the sample, and `logr` puts that on
        an additive scale. `v` is the same quantity as `sigma_x` from section 6,
        just measured relative to the object's own brightness instead of to the
        band's depth.
        """
    )
    return


@app.cell
def _(BANDS, BAND_NAMES, BLOCK_ORDER, alt, g, logr, mask, mo, np, pd, v):
    sampling_rng = np.random.default_rng(0)
    pixels = sampling_rng.choice(mask.size, size=min(18_000, mask.size), replace=False)
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
            x=alt.X("g:Q", title="g = f / r (shape, unit norm over measured bands)"),
            y=alt.Y("v:Q", title="v = sigma_g (shape error)", scale=alt.Scale(type="log")),
            tooltip=["g", "v"],
        )
        .properties(height=300, width=430, title="the noise model, cell by cell")
    )

    band_frame = pd.DataFrame(
        {"band": list(BAND_NAMES), "wavelength_nm": [b.wavelength_aa / 10.0 for b in BANDS]}
    )
    band_frame["median_g"] = np.array(
        [np.median(g[mask[:, j] == 1, j]) for j in range(len(BANDS))]
    )
    band_frame["lo"] = np.array(
        [np.percentile(g[mask[:, j] == 1, j], 16) for j in range(len(BANDS))]
    )
    band_frame["hi"] = np.array(
        [np.percentile(g[mask[:, j] == 1, j], 84) for j in range(len(BANDS))]
    )
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
        .properties(height=300, width=430, title="median shape across the sample")
    )
    band_frame_band = (
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
                ### The two channels, side by side

                Left: every measured cell of a 200k random sample, in the plane the
                likelihood lives in. The floor at `v ~ 0.005` is the relative
                calibration term; the cloud to the left is the noise-dominated
                regime, where `|g|` is small and `v` is large.

                Right: the sample's median SED shape, with the 16-84 percentile band.
                This is the "average object" a decoder that ignored its latent would
                have to produce - worth keeping in mind when judging any
                reconstruction (section 12).
                """
            ),
            mo.hstack([dots, mean_sed + band_frame_band], justify="space-around"),
        ]
    )
    return


@app.cell
def _(g, mask, mo, np, object_id, objects, v):
    detection = np.divide(
        np.abs(g.astype(np.float64)),
        v.astype(np.float64),
        out=np.zeros_like(g, dtype=np.float64),
        where=v > 0,
    )
    n_detected = (detection >= 5).sum(axis=1)
    n_measured = (mask == 1).sum(axis=1)
    sp_class = objects["spCLASSsdss"].to_numpy()
    log_r = np.log(np.sqrt((g.astype(np.float64) ** 2 * mask).sum(axis=1)) + 1e-12)

    candidates: dict[str, int] = {}
    for cls in ("GALAXY", "STAR", "QSO"):
        members = np.flatnonzero(sp_class == cls)
        if members.size:
            candidates[f"{cls} - brightest ({n_measured[members].max()} bands measured)"] = int(
                members[np.argmax(n_detected[members])]
            )
    candidates["median object - typical number of detections"] = int(
        np.argsort(-n_detected)[len(n_detected) // 2]
    )
    candidates["noisiest - fewest bands with a measured error"] = int(np.argmin(n_measured))
    candidates["faintest in r"] = int(np.argmin(log_r))

    options = {
        f"{object_id[i]} - {label} | {n_detected[i]} bands at |g/v| >= 5": i
        for label, i in candidates.items()
    }
    picker = mo.ui.dropdown(options=options, value=next(iter(options)), label="object")
    return detection, n_detected, picker


@app.cell
def _(
    BANDS,
    BAND_NAMES,
    alt,
    b_j_values,
    detection,
    g,
    logr,
    mask,
    mo,
    n_detected,
    np,
    object_id,
    objects,
    pd,
    picker,
    v,
):
    chosen = picker.value
    row = np.arange(len(BAND_NAMES))
    in_g = g[chosen].astype(np.float64)
    in_v = v[chosen].astype(np.float64)
    measured = mask[chosen] == 1
    r_object = float(np.exp(logr[chosen]))

    frame = pd.DataFrame(
        {
            "band": list(BAND_NAMES),
            "channel": row,
            "wavelength_nm": [b.wavelength_aa / 10.0 for b in BANDS],
            "g": in_g,
            "v": np.where(measured, in_v, np.nan),
            "flux": in_g * r_object,
            "measured": measured,
        }
    )

    def _chart(y: str, title: str, zero: bool = True) -> alt.LayerChart:
        base = alt.Chart(frame).encode(
            x=alt.X("wavelength_nm:Q", title="wavelength [nm]"),
            tooltip=["band", "measured", alt.Tooltip(f"{y}:Q", format=".4g")],
        )
        points = base.mark_point(size=18).encode(
            y=alt.Y(f"{y}:Q", title=title),
            color=alt.Color("measured:N", title="mask", scale=alt.Scale(domain=[True, False], range=["#4c78a8", "#e45756"])),
        )
        bars = base.transform_filter(alt.datum.measured).mark_rule(opacity=0.45).encode(
            y=f"{y}:Q", y2="v:Q"
        )
        chart = points
        if zero:
            chart = alt.layer(
                alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[3, 3], opacity=0.4).encode(y="y:Q"),
                points,
            )
        return (chart + bars).properties(height=200, width=620, title=title)

    shape_chart = _chart("g", f"{object_id[chosen]}: g (shape) with +-v")
    flux_chart = _chart("flux", "same object in raw flux units (g * r)")

    spectral = objects["spCLASSsdss"][chosen]
    photo_z = objects["zph"][chosen]
    spec_z = objects["zsp"][chosen]
    class_text = f"`{spectral}`" if pd.notna(spectral) else "no spectroscopic class"
    z_text = ", ".join(
        part
        for part in (
            f"`zsp = {spec_z:.3f}`" if pd.notna(spec_z) else "",
            f"`zph = {photo_z:.3f}`" if pd.notna(photo_z) else "",
        )
        if part
    ) or "no redshift"
    mo.vstack(
        [
            mo.md(
                """
                ## 8. One object: exactly what the network sees

                Every dot is one of the 60 channels; the bars are `+-v`, the scale
                the likelihood uses. Red marks a masked band - present in the tensor
                as `g = 0, v = 0`, contributing nothing to the loss.

                The top panel is the **decoder target**. The bottom panel multiplies
                it back by the object's own `r = exp(logr)`, which is the only thing
                that makes an SED look like a spectrum instead of a shape.
                """
            ),
            picker,
            shape_chart,
            flux_chart,
            mo.md(
                f"""
                `{object_id[chosen]}` is {class_text} with {z_text} - labels for
                interpretation only, never inputs. It has
                **{int(measured.sum())}/{len(BAND_NAMES)} measured bands**,
                **{int(n_detected[chosen])}** with `|g/v| >= 5`, `logr = {float(logr[chosen]):.2f}`
                (`r = {r_object:.1f}` in flux units), and
                **{int((~measured).sum())} masked** channel(s).
                """
            ),
        ]
    )
    return


@app.cell
def _(
    BANDS,
    BAND_NAMES,
    BLOCK_ORDER,
    alt,
    fold,
    g,
    logr,
    mask,
    mo,
    np,
    object_id,
    pd,
):
    rng_grid = np.random.default_rng(7)
    fully_measured = np.flatnonzero((mask == 1).all(axis=1))
    sample = rng_grid.choice(fully_measured, size=12, replace=False)
    long = pd.DataFrame(
        {
            "object_id": np.repeat([object_id[i] for i in sample], len(BAND_NAMES)),
            "band": list(BAND_NAMES) * len(sample),
            "wavelength_nm": np.tile([b.wavelength_aa / 10.0 for b in BANDS], len(sample)),
            "g": g[sample].ravel().astype(np.float64),
            "block": [b.block for b in BANDS] * len(sample),
            "logr": np.repeat(logr[sample], len(BAND_NAMES)).astype(np.float64),
        }
    )
    grid = (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=8))
        .encode(
            x=alt.X("wavelength_nm:Q", title="wavelength [nm]"),
            y=alt.Y("g:Q", title="g"),
            color=alt.Color("block:N", sort=list(BLOCK_ORDER), legend=None),
            tooltip=["object_id", "band", alt.Tooltip("g:Q", format=".3f")],
            facet=alt.Facet("object_id:N", columns=4, title=None),
        )
        .properties(width=150, height=85)
    )

    many = rng_grid.choice(fully_measured, size=300, replace=False)
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
            color=alt.Color(
                "logr:Q", title="log r", scale=alt.Scale(scheme="viridis")
            ),
        )
        .properties(height=300, width=620, title="300 random objects, coloured by brightness")
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 9. Many objects: is the shape channel actually varied?

                Twelve complete spectra (all 60 bands measured). Below them, 300
                objects overlaid: the vertical spread at each wavelength is the SED
                diversity the model has to encode, and the colour is brightness.

                Read the colour as a control: if brightness leaked into `g`, the
                curves would be ordered by colour. They are not, because `g` is
                unit-norm by construction - two objects of the same class at
                different distances land on the same curve. That is the reason
                `logr` is a separate target rather than a 61st channel.
                """
            ),
            grid,
            spaghetti,
        ]
    )
    return


@app.cell
def _(fold, g, logr, mask, mo, np):
    measured_x = np.where(mask == 1, g, np.nan).astype(np.float64)
    level = np.nanmean(measured_x, axis=1, keepdims=True)
    shape = measured_x - level
    total = np.nanvar(measured_x)
    mo.md(
        f"""
        ### A warning that shapes every later result

        Split the shape channel into an overall level and a per-band deviation
        from it:

        | component | share of variance in `g` |
        |---|---|
        | per-object level (mean of `g`) | **{100 * np.nanvar(level) / total:.0f}%** |
        | per-band shape (colour) | **{100 * np.nanvar(shape) / total:.0f}%** |

        A large fraction of `g`'s variance is still a single number per object: how
        much of the total flux sits in the mean band versus the edges. That is why
        a very small latent can already reconstruct a lot, and why the reconstruction
        number must always be compared with the "predict the sample mean" baseline
        (section 12) rather than taken at face value.
        """
    )
    return


@app.cell
def _(BAND_NAMES, alt, mask, mo, np, pd):
    per_band = pd.DataFrame(
        {"band": list(BAND_NAMES), "masked_fraction": 1.0 - mask.mean(axis=0)}
    )
    chart = (
        alt.Chart(per_band)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=list(BAND_NAMES), title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("masked_fraction:Q", title="fraction of objects masked", axis=alt.Axis(format="%")),
            tooltip=["band", alt.Tooltip("masked_fraction:Q", format=".4f")],
        )
        .properties(height=200)
    )
    per_object = pd.Series((mask == 0).sum(axis=1)).value_counts().sort_index()
    n_any = int((mask == 0).any(axis=1).sum())
    n_all = int((mask == 0).all(axis=1).sum())
    worst_band = per_band.loc[per_band["masked_fraction"].idxmax()]
    mo.vstack(
        [
            mo.md(
                f"""
                ## 10. The mask channel

                `mask` is 1 where the export quoted a non-zero relative error. It is
                an input channel as well as a loss selector, so the model can tell
                "unmeasured band" from "measured, flux near zero" - without it those
                two look identical after `g = 0`.

                * {n_any:,} objects have at least one masked channel
                * {n_all} objects are fully masked
                * {int((mask == 0).sum()):,} pixels ({100 * (mask == 0).mean():.2f}%)
                * worst band: `{worst_band["band"]}` at {100 * worst_band["masked_fraction"]:.2f}%

                Distribution over objects: {", ".join(f"{k} masked -> {v:,} obj" for k, v in per_object.head(4).items())} ...

                The loss multiplies by `mask`, so a masked cell has zero weight no
                matter what the decoder predicts. Nothing imputes these pixels.
                """
            ),
            chart,
        ]
    )
    return


@app.cell
def _(fold, mo, np, tile_id):
    rows = []
    for k in np.unique(fold):
        held_tile = np.unique(tile_id[fold == k])
        rows.append(
            f"| {k} | {', '.join(str(t) for t in held_tile)} | "
            f"{int((fold == k).sum()):,} | {int((fold != k).sum()):,} |"
        )
    fold_table = (chr(10) + " " * 8).join(rows)
    mo.md(
        f"""
        ## 11. Splits: leave one pointing out

        | fold | held-out tile | validation objects | training objects |
        |---|---|---|---|
        {fold_table}

        The unit of independence is the pointing, not the object. Objects are never
        shared between the four miniJPAS pointings, but neighbours inside one
        pointing share sky background, crowding and a common photometric
        calibration. A random row split would train and score on the same noise.

        The pointings also differ in depth, so a fold score is partly a
        depth-mismatch score. That is the intended reading - it is the production
        situation, applying a model to a new field.

        ```python
        train, val = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        ```
        """
    )
    return


@app.cell
def _(BAND_NAMES, alt, fold, mo, np, pd, v):
    train = fold != 0
    floor = np.array(
        [
            np.quantile(v[train, j][v[train, j] > 0], 0.05)
            for j in range(len(BAND_NAMES))
        ]
    )
    median_v = np.array([np.median(v[train, j][v[train, j] > 0]) for j in range(len(BAND_NAMES))])
    floor_frame = pd.DataFrame(
        {
            "band": list(BAND_NAMES) * 2,
            "value": np.concatenate([floor, median_v]),
            "kind": ["floor s_j (p5 of v, train fold)"] * len(BAND_NAMES)
            + ["median v (train fold)"] * len(BAND_NAMES),
        }
    )
    floor_chart = (
        alt.Chart(floor_frame)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=list(BAND_NAMES), title=None, axis=alt.Axis(labelAngle=-90)),
            xOffset=alt.XOffset("kind:N"),
            y=alt.Y("value:Q", title="v (shape-error units)"),
            color=alt.Color("kind:N", title=None),
            tooltip=["band", "kind", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=220)
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 12. The noise floor, and why `v` alone is not enough

                The archive's error is **purely relative**, so `v` shrinks with the
                flux and a near-zero-flux band would claim infinite precision. The
                pipeline therefore adds a fixed per-band floor `s_j`, a low
                percentile of `v` computed **on the training rows of the current
                fold only** and then frozen:

                ```python
                s_j = percentile(v[mask == 1, j][train_rows], {5.0:g})
                w   = sqrt(v^2 + s_j^2)      # the variance the likelihood uses
                ```

                Here `s_j` ranges {floor.min():.4f} .. {floor.max():.4f}, i.e.
                {100 * float(np.median(floor / median_v)):.0f}% of a typical band's
                `v`. It is never learned: it is a property of the instrument, and
                fitting it on all folds would leak the held-out tile's noise level
                into the model.

                Note also that the floor is *not* in `vae_inputs.npz`. That is
                deliberate - a stored floor would be a floor fitted on every object,
                including the held-out tile. It is written into `models/vae.pt` after
                training so `predict.py` uses the same one.
                """
            ),
            floor_chart,
        ]
    )
    return


@app.cell
def _(MODELS_DIR, REPORTS_DIR, mo, np):
    import json
    import torch

    from jpas_da.modeling.vae import build_model, normalise_shape

    model_path = MODELS_DIR / "vae.pt"
    metrics_path = REPORTS_DIR / "metrics.json"
    if not model_path.exists():
        diagnostics = None
        mo.md(f"`{model_path}` not found - run `make train` or `make train-debug`.")
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        model = build_model(checkpoint["model"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        from jpas_da.modeling import data as _model_data

        a = _model_data.load(checkpoint["config"]["data"]["path"])
        inputs = np.stack([a[name].astype(np.float32) for name in checkpoint["channels"]], axis=1)
        m = a["mask"].astype(np.float32)
        with torch.no_grad():
            out = model(torch.from_numpy(inputs))
            mu = normalise_shape(out["mu_g"], torch.from_numpy(m)).numpy()

        g_arr, mask_arr = a["g"], a["mask"]
        val_mask = a["fold"] == checkpoint["config"]["data"]["fold"]

        def _rmse(pred: np.ndarray, idx: np.ndarray) -> float:
            resid = (pred[idx] - g_arr[idx]) ** 2 * mask_arr[idx]
            return float(np.sqrt(resid.sum(axis=1) / mask_arr[idx].sum(axis=1)).mean())

        train_mean = g_arr[~val_mask].mean(axis=0)
        train_mean = train_mean / np.linalg.norm(train_mean)
        diagnostics = {
            "debug": bool(metrics.get("debug")),
            "epochs": int(metrics.get("final", {}).get("epoch", 0)),
            "channels": list(checkpoint["channels"]),
            "latent_dim": int(checkpoint["model"]["latent_dim"]),
            "val_rmse": _rmse(mu, val_mask),
            "train_rmse": _rmse(mu, ~val_mask),
            "mean_rmse": _rmse(np.broadcast_to(train_mean, g_arr.shape), val_mask),
            "latent_std": out["mu_z"].std(dim=0).numpy(),
            "latent_logr_corr": [
                float(np.corrcoef(out["mu_z"][:, k].numpy(), a["logr"])[0, 1])
                for k in range(out["mu_z"].shape[1])
            ],
        }
    return (diagnostics,)


@app.cell
def _(diagnostics, mo, np):
    if diagnostics is None:
        mo.md("")
    else:
        run_kind = "**debug** run (`make train-debug`)" if diagnostics["debug"] else "full run"
        corr = ", ".join(
            f"`z_{k}` {c:+.2f}" for k, c in enumerate(diagnostics["latent_logr_corr"])
        )
        beats = diagnostics["val_rmse"] < diagnostics["mean_rmse"]
        verdict = (
            "better than the mean-shape baseline"
            if beats
            else "**worse than the mean-shape baseline**"
        )
        note = (
            "Read this as a pipeline check, not a result: two epochs cannot beat a\n"
            "one-parameter baseline, and the near-flat latent spread says the\n"
            "posterior has barely moved off its prior. If this cell shows the decoder\n"
            "*ahead* of the baseline after a real `make train`, the setup is sound."
            if diagnostics["debug"]
            else "A full run: this is where the model's quality can be judged."
        )
        output = mo.md(
            f"""
            ### Checking the checkpoint against the data it was trained on

            `models/vae.pt` currently on disk is a {run_kind},
            {diagnostics["epochs"]} epoch(s), channels `{diagnostics["channels"]}`,
            latent width {diagnostics["latent_dim"]}.

            | | masked RMSE of `g` |
            |---|---|
            | decoder, training folds | {diagnostics["train_rmse"]:.4f} |
            | decoder, held-out fold | {diagnostics["val_rmse"]:.4f} |
            | "predict the training-mean shape", held-out fold | {diagnostics["mean_rmse"]:.4f} |

            The decoder is {verdict}. Latent spread per dimension:
            {", ".join(f"{s:.3f}" for s in diagnostics["latent_std"])}.

            Correlation of each latent with brightness: {corr}.

            {note}
            """
        )
    output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. What must not go in, and the guarantees the pipeline makes

    | do not feed | why |
    |---|---|
    | `MAGrSDSS` | exactly `26.1147 - 2.5*log10(rSDSS)`; a duplicate of a channel |
    | `angDist` | non-null exactly on the SDSS-cross-matched rows: encodes label availability |
    | `zsp`, `zph`, `avg_zph`, `e_*` | the labels the model is meant to help predict |
    | `spCLASSsdss`, `photoCLASSsdss` | same |
    | `ALPHA_J2000`, `DELTA_J2000`, `flag_rSDSS` | position and selection, not photometry |

    All of them stay in `objects.parquet` so downstream inference can join them
    back on `object_id`.

    | requirement | enforced by |
    |---|---|
    | no NaN in any model channel | `model_data.normalise` raises on non-finite `g`/`v`/`logr` |
    | `v == 0` exactly where `mask == 0` | constructed that way, asserted in `tests/test_data.py` |
    | `g` unit-norm over measured bands | `np.sqrt((g^2 * mask).sum(1)) == 1`, checked above |
    | canonical band order | one definition, `jpas_da.bands.BAND_NAMES` |
    | exact invertibility of the asinh view | `features.check_roundtrip` on every build |
    | fold assignment covers every object | `model_data.build` raises otherwise |
    | floor fitted on training rows only | `train.noise_floor` receives `train_index` |
    | checkpoint cannot be loaded into a different architecture | `build_model` takes every value explicitly from the checkpoint |

    ### Reading the rest of the project

    * `vae-normalisation-and-loss.md` - derivations and measurements behind the
      normalisation and each loss term.
    * `jpas_da/modeling/data.py` - the RMS decomposition, in ~50 lines.
    * `jpas_da/modeling/train.py` - floor, KL warm-up, brightness head, checkpoint.
    * `data/processed/vae_latents.parquet` - what `make predict` writes; start any
      analysis from there rather than re-running the network.
    """)
    return


if __name__ == "__main__":
    app.run()
