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
    from jpas_da.bands import BANDS, BAND_NAMES, BLOCK_ORDER
    from jpas_da.config import INTERIM_DATA_DIR

    return (
        BANDS,
        BAND_NAMES,
        BLOCK_ORDER,
        INTERIM_DATA_DIR,
        alt,
        features,
        mo,
        np,
        pd,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The training data: what the VAE sees

    Everything below reads the pipeline artifacts, never the 75 MB CSV:

    | Artifact | Built by | Holds |
    |---|---|---|
    | `data/interim/objects.parquet` | `make data` | one row per object: raw flux, absolute sigma, validity mask, metadata |
    | `data/interim/band_scales.parquet` | `make data` | per-band soft scale `b_j` |
    | `data/processed/jpas_asinh.parquet` | `make features` | the model input `x`, its uncertainty `sigma_x`, the mask |
    | `data/interim/folds.parquet` | `make splits` | leave-one-tile-out fold per object |

    The model is trained on **flux and uncertainty only**. Labels (`zsp`,
    `zph`, `spCLASSsdss`, `photoCLASSsdss`) and position/metadata columns are
    carried alongside for inference and diagnostics, never fed in.
    """)
    return


@app.cell
def _(BAND_NAMES, INTERIM_DATA_DIR, features, mo, pd):
    arrays = features.load_arrays()
    objects = pd.read_parquet(INTERIM_DATA_DIR / "objects.parquet")
    scales = pd.read_parquet(INTERIM_DATA_DIR / "band_scales.parquet")

    n_objects = arrays["x"].shape[0]
    n_bands = len(BAND_NAMES)
    n_masked_pixels = int((arrays["mask"] == 0).sum())
    n_complete = int((arrays["mask"] == 1).all(axis=1).sum())

    mo.md(
        f"""
        ## 1. One row per object

        `load_arrays()` returns aligned numpy arrays plus identifiers. Nothing
        else is needed to start training.

        | | |
        |---|---|
        | objects | **{n_objects:,}** |
        | bands per object | **{n_bands}** |
        | array shapes | `x`, `sigma_x`, `mask` = `({n_objects:,}, {n_bands})` |
        | objects with all {n_bands} uncertainties measured | {n_complete:,} |
        | pixels with no measured uncertainty | {n_masked_pixels:,} ({100 * n_masked_pixels / arrays["mask"].size:.2f}%) |
        | folds | {len(set(arrays["fold"].tolist()))} |
        """
    )
    return arrays, scales


@app.cell
def _(arrays):
    x = arrays["x"]
    sigma_x = arrays["sigma_x"]
    mask = arrays["mask"]
    fold = arrays["fold"]
    object_id = arrays["object_id"]
    tile_id = arrays["tile_id"]
    return fold, mask, sigma_x, tile_id, x


@app.cell
def _(np, scales):
    b_j_values = scales["b_j"].to_numpy(dtype=np.float64)
    return (b_j_values,)


@app.cell
def _(BAND_NAMES, mo):
    mo.md(f"""
    ### How to load it

    ```python
    from jpas_da.features import load_arrays

    a = load_arrays()          # -> dict of arrays
    x, sigma_x, mask = a["x"], a["sigma_x"], a["mask"]
    fold = a["fold"]           # leave-one-tile-out assignment
    ```

    `x` and `sigma_x` are `float32` of shape `(n_objects, {len(BAND_NAMES)})`,
    `mask` is `int8`. **Band order is canonical and identical in all three
    arrays** - it comes from `jpas_da.bands.BAND_NAMES`, so no column matching
    is needed and no ordering can drift.
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
                "width_nm": None if b.width_aa is None else b.width_aa / 10.0,
                "index": i,
            }
            for i, b in enumerate(BANDS)
        ]
    )
    band_chart = (
        alt.Chart(band_table)
        .mark_tick(size=16, thickness=3)
        .encode(
            x=alt.X("wavelength_nm:Q", title="wavelength [nm]"),
            y=alt.Y("index:Q", title="channel index", axis=alt.Axis(tickCount=8)),
            color=alt.Color("block:N", title="block", sort=list(BLOCK_ORDER)),
            tooltip=["band", "block", "wavelength_nm", "width_nm", "index"],
        )
        .properties(height=320, title="60 channels: the 54 narrow bands are the spectrum")
    )
    counts = band_table["block"].value_counts().reindex(list(BLOCK_ORDER))
    mo.vstack(
        [
            mo.md(
                """
                ## 2. The 60 channels

                | block | n | members | why it is in the tensor |
                |---|---|---|---|
                | narrow | 54 | `J0378` ... `J0910` | the J-spectrum: a 10 nm grid (`J0378`->`J0390` is 12 nm), FWHM ~14.5 nm |
                | medium | 1 | `uJAVA` | FWHM 49.5 nm centred at 350 nm, the blue anchor |
                | broad | 4 | `uJPAS`, `gSDSS`, `rSDSS`, `iSDSS` | `rSDSS` is the detection band and ~4.6x deeper than a narrow band |
                | highpass | 1 | `J1007` | the red edge |
                """
            ),
            mo.hstack(
                [
                    mo.stat(value=f"{int(counts[b])}", label=f"{b} bands", bordered=True)
                    for b in counts.index
                ]
            ),
            band_chart,
        ]
    )
    return


@app.cell
def _(alt, b_j_values, mask, mo, np, pd, sigma_x, x):
    rng = np.random.default_rng(0)
    sample = rng.choice(x.size, size=min(300_000, x.size), replace=False)
    x_flat = x.ravel()[sample]
    flux_flat = np.sinh(x_flat / 2.0)  # presentation factor only; b_j is per band
    signed_flux = np.sign(flux_flat) * np.log10(np.abs(flux_flat) + 1)

    def histogram(values: np.ndarray, bins: int = 80) -> pd.DataFrame:
        counts, edges = np.histogram(values, bins=bins)
        return pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})

    hist_x = (
        alt.Chart(histogram(x_flat))
        .mark_bar()
        .encode(
            x=alt.X("left:Q", title="asinh coordinate x"),
            x2="right:Q",
            y=alt.Y("count:Q", title="pixels", scale=alt.Scale(type="log")),
        )
        .properties(height=200, width=330, title="x = asinh(f/b) * 2 - the model input")
    )
    hist_flux = (
        alt.Chart(histogram(signed_flux))
        .mark_bar()
        .encode(
            x=alt.X("left:Q", title="sign(f) * log10(|f| + 1)"),
            x2="right:Q",
            y=alt.Y("count:Q", title="pixels", scale=alt.Scale(type="log")),
        )
        .properties(height=200, width=330, title="raw flux, same pixels (log-signed axis)")
    )
    negative_share = float((x[mask == 1] < 0).mean())
    # SNR must be measured in flux space: |x|/sigma_x is *not* SNR, because the
    # asinh squashes the bright end by a factor asinh(f/b)/(f/b).
    flux_derived = b_j_values * np.sinh(x / 2.0)
    sigma_derived = sigma_x.astype(np.float64) * np.sqrt(flux_derived**2 + b_j_values**2) / 2.0
    flux_snr = np.divide(
        np.abs(flux_derived),
        sigma_derived,
        out=np.full_like(flux_derived, np.nan),
        where=(mask == 1) & (sigma_derived > 0),
    )
    low_snr_share = float(np.nanmean(flux_snr < 1))
    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. Why the channels are not raw flux

                Two facts from the export force the transform:

                * **{100 * negative_share:.1f}% of pixels have negative flux** and
                  **{100 * low_snr_share:.0f}% of measured pixels have |SNR| < 1**. A logarithm
                  is undefined on a fifth of the data and would need an arbitrary
                  floor.
                * The **brightest 0.1% of pixels carry 99.9% of** `sum(f^2)`. An
                  L2 loss on raw flux is a least-squares fit to a few thousand pixels;
                  the other 64,000 objects are invisible to it.

                `asinh` is linear while `|f| < b` (noise regime, negatives kept as
                negatives) and logarithmic for `|f| >> b` (the four-decade bright
                tail), and it is exactly invertible. There is no imputation and no
                clipping anywhere in the pipeline.
                """
            ),
            mo.hstack([hist_x, hist_flux], justify="space-around"),
        ]
    )
    return


@app.cell
def _(BAND_NAMES, alt, mo, scales):
    scale_chart = (
        alt.Chart(scales)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=list(BAND_NAMES), title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("b_j:Q", title="b_j = median |sigma_f|"),
            tooltip=["band", alt.Tooltip("b_j:Q", format=".2f"), "n_sigma_used"],
        )
        .properties(height=240, title="per-band soft scale b_j, in raw flux units")
    )
    by_band = scales.set_index("band")["b_j"]
    depth_ratio = float(by_band["uJAVA"] / by_band["rSDSS"])
    mo.vstack(
        [
            mo.md(
                f"""
                ## 4. The normalisation, in full

                Two numbers are applied to the raw flux, and both are stored so the
                transform is reversible without guessing.

                **Step 1 - the soft scale `b_j`, one per band.**
                `b_j` is the median absolute flux uncertainty of that band
                (`data/interim/band_scales.parquet`). It is the flux at which a band
                stops being noise-dominated, so it is the natural pivot:

                | band | `b_j` (flux units) |
                |---|---|
                | `uJAVA` | {by_band['uJAVA']:.2f} |
                | `gSDSS` | {by_band['gSDSS']:.2f} |
                | `rSDSS` | {by_band['rSDSS']:.2f} |
                | `iSDSS` | {by_band['iSDSS']:.2f} |

                The broad SDSS bands have `b_j` **{depth_ratio:.0f}x smaller** than `uJAVA`:
                they are far deeper. `b_j` absorbs that per-band depth difference
                instead of one global scale trying to fit all 60 bands.

                It is computed **globally, not per tile**, on purpose. The four
                pointings differ in depth by 0.16 dex typically (0.37 dex in the worst
                bands). Normalising that away would hide a real systematic that
                leave-one-tile-out exists to measure.
                """
            ),
            scale_chart,
        ]
    )
    return


@app.cell
def _(BAND_NAMES, INTERIM_DATA_DIR, b_j_values, features, mask, mo, np, pd, x):
    interim_flux = pd.read_parquet(
        INTERIM_DATA_DIR / "objects.parquet", columns=[f"flux_{b}" for b in BAND_NAMES]
    )
    flux_truth = interim_flux.to_numpy(dtype=np.float64)
    recovered = features.asinh_inverse(x / features.PRESENTATION_SCALE, b_j_values)
    # Only measured pixels can invert: a masked pixel stores x = 0 by design.
    measured = mask == 1
    relative_residual = np.abs(recovered - flux_truth)[measured] / np.maximum(
        np.abs(flux_truth[measured]), 1e-12
    )

    mo.md(
        f"""
        ### The transform

        ```python
        x_b     = asinh(f_b / b_b) * 2                     # model input
        sigma_x = sigma_f / sqrt(f_b**2 + b_b**2) * 2      # propagated uncertainty
        f_b     = b_b * sinh(x_b / 2)                      # exact inverse
        ```

        | quantity | source | note |
        |---|---|---|
        | `f_b` | `flux_<band>` in the interim table | `FLUX_AUTO`, zeropoint 26.1147 (these are **not** uJy) |
        | `sigma_f` | `abs(flux * FLUX_RELERR_AUTO)` | the export's error is *signed relative*, so it must be multiplied by the flux |
        | `b_b` | `b_j` per band | median `sigma_f` |
        | `2` | `features.PRESENTATION_SCALE` | monotonic global factor that only sets the numerical size of the channels and changes no distance ordering |

        The `2` matters less than it looks: `asinh(x)/s` is monotonic, so any
        analysis that standardises, whitens or computes distances is invariant to
        it. It is kept in the file so the inverse stays exact.

        `check_roundtrip` asserts `x -> f` on every build. Re-running it on the
        shipped file: over the {int(measured.sum()):,} measured pixels the **maximum
        relative error is `{relative_residual.max():.1e}`** (median
        `{np.median(relative_residual):.1e}`), which is the float32 storage limit.
        A model output therefore maps back to physical flux with no approximation
        step. Masked pixels are excluded and cannot invert: their `x` is 0 by
        construction, which is exactly why they must be excluded from the loss
        rather than reconstructed.
        """
    )
    return


@app.cell
def _(BAND_NAMES, INTERIM_DATA_DIR, alt, mask, mo, np, pd, sigma_x):
    interim_sigma = pd.read_parquet(
        INTERIM_DATA_DIR / "objects.parquet", columns=[f"sigma_{b}" for b in BAND_NAMES]
    )
    interim_flux_full = pd.read_parquet(
        INTERIM_DATA_DIR / "objects.parquet", columns=[f"flux_{b}" for b in BAND_NAMES]
    )

    decile = pd.qcut(np.abs(interim_flux_full.to_numpy()).ravel(), 10, labels=False, duplicates="drop")
    sigma_f = interim_sigma.to_numpy(dtype=np.float64)
    measured_x = np.where(mask == 1, sigma_x.astype(np.float64), np.nan)

    median_flux_units = pd.Series(sigma_f.ravel()).groupby(decile).median()
    median_asinh_units = pd.Series(measured_x.ravel()).groupby(decile).median()
    comparison = pd.DataFrame(
        {
            "decile": list(median_flux_units.index) * 2,
            "median_sigma": list(median_flux_units.to_numpy())
            + list(median_asinh_units.to_numpy()),
            "space": ["flux units"] * len(median_flux_units) + ["asinh units"] * len(median_asinh_units),
        }
    )
    flatness = (
        alt.Chart(comparison)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "decile:Q",
                title="decile of |flux| (0 = faintest)",
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y("median_sigma:Q", title="median sigma", scale=alt.Scale(type="log")),
            color=alt.Color("space:N", title=None),
            tooltip=["space", "decile", alt.Tooltip("median_sigma:Q", format=".3g")],
        )
        .properties(height=260, width=560, title="the noise, in the output coordinate")
    )
    span_flux = float(median_flux_units.max() / median_flux_units.min())
    faint_x = median_asinh_units.iloc[:-1]
    span_x = float(faint_x.max() / faint_x.min())
    drop_bright = float(faint_x.median() / median_asinh_units.iloc[-1])
    mo.vstack(
        [
            mo.md(
                f"""
                ## 5. The uncertainty survives the transform

                A heteroscedastic Gaussian likelihood needs `sigma` in the *same*
                coordinate as the data, so the error is propagated by the chain rule,
                `d asinh(f/b)/df = 1/sqrt(f^2 + b^2)`.

                That is the property which makes this representation worth the trouble:

                * in **flux units** the median sigma grows **{span_flux:.1f}x** across flux deciles;
                * in **asinh units** it spans only **{span_x:.1f}x over the nine faintest deciles**,
                  and the final collapse (**{drop_bright:.0f}x** lower) is real - the brightest objects
                  genuinely are measured better.

                So the model does not need a separate "trust bright pixels more"
                mechanism: the `sigma` channel already says it.
                """
            ),
            flatness,
        ]
    )
    return


@app.cell
def _(INTERIM_DATA_DIR, mo, np, pd, sigma_x, x):
    labels = pd.read_parquet(
        INTERIM_DATA_DIR / "objects.parquet", columns=["object_id", "spCLASSsdss", "zph"]
    )
    detection_snr = np.divide(
        np.abs(x), sigma_x, out=np.zeros_like(x), where=np.isfinite(sigma_x)
    )
    n_detected = (detection_snr >= 5).sum(axis=1)
    ranked = np.argsort(-n_detected)
    # Prefer an object that carries a spectroscopic class, so the panel has
    # something to describe; only 1,168 of 64,293 objects are annotated.
    annotated = labels["spCLASSsdss"].notna().to_numpy()
    brightest_annotated = next(
        (int(i) for i in ranked if annotated[i]),
        int(ranked[0]),
    )
    median_object = int(ranked[len(ranked) // 2])

    choices = {}
    for candidate, description in (
        (brightest_annotated, "brightest annotated"),
        (median_object, "median number of detections"),
    ):
        choices[
            f"{labels['object_id'][candidate]} - {description} "
            f"({n_detected[candidate]} bands at SNR>=5)"
        ] = candidate
    picker = mo.ui.dropdown(options=choices, value=next(iter(choices)), label="object")
    return labels, picker


@app.cell
def _(
    BAND_NAMES,
    alt,
    b_j_values,
    labels,
    mask,
    mo,
    np,
    pd,
    picker,
    sigma_x,
    x,
):
    chosen = picker.value
    in_x = x[chosen].astype(np.float64)
    object_frame = pd.DataFrame(
        {
            "band": list(BAND_NAMES),
            "channel": np.arange(len(BAND_NAMES)),
            "x": in_x,
            "sigma_x": sigma_x[chosen].astype(np.float64),
            "flux": b_j_values * np.sinh(in_x / 2.0),
            "masked": mask[chosen] == 0,
        }
    )
    base = alt.Chart(object_frame).encode(
        x=alt.X(
            "channel:Q",
            title="channel index (canonical band order)",
            axis=alt.Axis(tickCount=10),
        ),
        tooltip=[
            "band",
            "masked",
            alt.Tooltip("x:Q", format=".3f"),
            alt.Tooltip("sigma_x:Q", format=".3f"),
            alt.Tooltip("flux:Q", format=".2f"),
        ],
    )
    in_model_space = (
        base.mark_point(size=20)
        .encode(y=alt.Y("x:Q", title="x = asinh(f/b) * 2"))
        .properties(
            height=190,
            width=560,
            title=f"{labels['object_id'][chosen]} in model coordinates",
        )
    )
    error_bars = (
        base.transform_filter(alt.datum.masked == False)  # noqa: E712
        .mark_rule(opacity=0.5)
        .encode(y="x:Q", y2="sigma_x:Q")
    )
    in_flux_space = (
        base.mark_point(size=20)
        .encode(y=alt.Y("flux:Q", title="flux (raw units)"))
        .properties(height=190, width=560, title="the same object, back in flux units")
    )
    spectral_class = labels["spCLASSsdss"][chosen]
    photo_z = labels["zph"][chosen]
    class_text = f"`{spectral_class}`" if pd.notna(spectral_class) else "spectroscopically unclassified"
    redshift_text = (
        f"`zph = {photo_z:.3f}`" if pd.notna(photo_z) else "no SDSS photo-z"
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 6. One object, both ways

                The same source in model coordinates and in physical flux. The lower
                panel is the stored inverse applied to the stored `x`, not a
                re-derivation for display.

                Watch the error bars change shape: roughly constant in `x`, expanding
                with brightness in flux.
                """
            ),
            picker,
            in_model_space + error_bars,
            in_flux_space,
            mo.md(
                f"""
                This object is {class_text}, {redshift_text} (inference only, never in
                training), with {int((mask[chosen] == 0).sum())} masked channel(s).
                """
            ),
        ]
    )
    return


@app.cell
def _(BAND_NAMES, alt, mask, mo, pd):
    masked_per_band = pd.DataFrame(
        {"band": list(BAND_NAMES), "masked_fraction": 1.0 - mask.mean(axis=0)}
    )
    masked_chart = (
        alt.Chart(masked_per_band)
        .mark_bar()
        .encode(
            x=alt.X("band:N", sort=list(BAND_NAMES), axis=alt.Axis(labelAngle=-90), title=None),
            y=alt.Y(
                "masked_fraction:Q",
                title="fraction of objects masked",
                axis=alt.Axis(format="%"),
            ),
            tooltip=["band", alt.Tooltip("masked_fraction:Q", format=".4f")],
        )
        .properties(height=200, title="masked pixels per band")
    )
    masked_per_object = pd.Series((mask == 0).sum(axis=1)).value_counts().sort_index()
    any_masked = int((mask == 0).any(axis=1).sum())
    all_masked = int((mask == 0).all(axis=1).sum())
    total_masked = int((mask == 0).sum())
    mo.vstack(
        [
            mo.md(
                f"""
                ## 7. The mask channel

                `mask` is 1 where the uncertainty was measured, and **0 where the
                export stored `FLUX_RELERR_AUTO == 0`**. That zero means *missing*, not
                *infinitely precise*: those pixels sit on faint sources (72% below their
                object's median brightness), in the noisiest bands, and
                disproportionately in one pointing. Treating them as high SNR - which
                an `f/sigma` representation would do implicitly - is a silent, biased
                error.

                * {any_masked:,} objects have at least one masked channel
                * {all_masked} objects are fully masked
                * {total_masked:,} pixels in total ({100 * total_masked / mask.size:.2f}%)

                Convention: masked pixels carry `x = 0` and `sigma_x = NaN`. Zero is
                unambiguous as "no measurement" because real non-detections scatter
                around `|x| ~ 1`. Exclude them from the likelihood with `mask == 1`,
                and pass `mask` as an input channel so the model can distinguish an
                unmeasured band from a measured zero.
                """
            ),
            masked_chart,
            mo.md(
                "Masked channels per object: "
                + ", ".join(
                    f"{count:,} object(s) with {k}" for k, count in masked_per_object.head(6).items()
                )
                + " ..."
            ),
        ]
    )
    return


@app.cell
def _(mask, mo, np, x):
    usable_x = np.where(mask == 1, x, np.nan)
    per_object_level = np.nanmean(usable_x, axis=1, keepdims=True)
    sed_shape = usable_x - per_object_level
    total_variance = np.nanvar(usable_x)
    level_share = float(np.nanvar(per_object_level) / total_variance)
    shape_share = float(np.nanvar(sed_shape) / total_variance)

    mo.md(
        f"""
        ## 8. What the 60 numbers actually vary with

        Split each spectrum into an overall level and a shape (`x - mean(x)`):

        | component | share of total variance |
        |---|---|
        | per-object brightness (level) | **{100 * level_share:.0f}%** |
        | SED shape (colour) | **{100 * shape_share:.0f}%** |

        Brightness dominates. It is a distance/luminosity axis, so it carries real
        physics and is deliberately **not** normalised away.

        Two consequences:

        * **For the VAE**, most latent capacity and most reconstruction loss will go
          on brightness unless you condition on it or reserve an explicit latent
          dimension for it.
        * **For Gromov-Wasserstein**, the distance matrix is brightness-dominated. If
          the question is about SED *families*, factor brightness out at the distance
          step rather than by re-transforming the data - that keeps the stored
          representation invertible.
        """
    )
    return


@app.cell
def _(fold, mo, np, tile_id):
    table_rows = []
    for fold_index in np.unique(fold):
        held_out = np.unique(tile_id[fold == fold_index])
        table_rows.append(
            f"| {fold_index} | {', '.join(str(t) for t in held_out)} | "
            f"{int((fold == fold_index).sum()):,} | {int((fold != fold_index).sum()):,} |"
        )
    mo.md(
        f"""
        ## 9. Splits: leave one pointing out

        | fold | held-out tile | test objects | train objects |
        |---|---|---|---|
        {chr(10).join(table_rows)}

        Grouping by tile is not cosmetic. Objects are never shared between pointings,
        but neighbours *inside* one pointing share sky background and a common
        photometric calibration. A random row split would train and score on the same
        noise, and would report a number the survey cannot reproduce on a new field.

        ```python
        train, test = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        ```

        Because the pointings also differ in depth (~0.16 dex), a fold score is
        partly a depth-mismatch score. That is the intended reading, not an artefact.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Feeding it to the VAE

    ```python
    a = load_arrays()
    x, sigma_x, mask = a["x"], a["sigma_x"], a["mask"]
    train = a["fold"] != k

    # 60 flux channels + 60 mask channels -> 120 inputs
    inputs = np.concatenate([x[train], mask[train].astype("float32")], axis=1)

    # heteroscedastic Gaussian likelihood, measured pixels only
    valid = mask[train] == 1
    noise = sigma_x[train] ** 2
    loss = ((rec - x[train]) ** 2 / (2 * noise) + np.log(sigma_x[train]))[valid].mean()
    ```

    The pipeline already guarantees:

    | requirement | where it is enforced |
    |---|---|
    | finite inputs, no NaN in `x` | `features.build_representation` raises otherwise |
    | `sigma_x` defined exactly where `mask == 1` | asserted at build time and in `tests/test_data.py` |
    | canonical band order | `jpas_da.bands.BAND_NAMES`, a single definition |
    | exact invertibility | `features.check_roundtrip`, run on every build |
    | no label leakage | labels live in `objects.parquet`; `x` holds 60 flux channels and nothing else |
    | no train/test leak | `folds.parquet`, one fold per pointing |

    **Do not** feed `MAGrSDSS` (exactly `26.1147 - 2.5*log10(rSDSS)`), `angDist`
    (non-null exactly on the SDSS-cross-matched rows), or any of
    `zsp`/`zph`/`avg_zph`/`spCLASSsdss`/`photoCLASSsdss`. All are kept in the
    interim table for later inference; none belongs in `x`.
    """)
    return


if __name__ == "__main__":
    app.run()
