---
title: "miniJPAS fluxes into a VAE: normalisation and loss"
note: working note, not part of the site
scope: [normalisation, loss function]
status: draft
evidence: every number marked (measured) is reproduced below from data/interim/objects.parquet
---

> **Partly superseded (2026-11-06)** by `references/pipeline-v2-plan.md`. The
> normaliser is now the broad-band RMS `r_bb` (not the RMS over all bands with a
> unit-sphere target). The claim that the quoted error "vanishes where the flux
> vanishes" does not hold on this export: the extreme weight range comes from very
> bright objects, and the additive floor is now a fixed brightness-relative `ε`,
> not a fitted percentile. The measurements below remain valid as measurements.


# miniJPAS fluxes into a VAE: normalisation and loss

Working note for a variational autoencoder trained on the miniJPAS 60-band fluxes
and their errors, with the aim of embedding the spectral energy distribution.
It covers two choices only: how to put the fluxes on a well-behaved scale, and
what to minimise. Galactic dereddening is deliberately **not** applied (see
[Why no dereddening](#why-no-dereddening) for the consequences).

Citations use the citekeys of a paper collection that is **not** in this
repository: there is no `references.bib` here, so every `[@key]` below is an
unverified pointer to a paper read elsewhere, and the locators are the papers'
own section, equation, table and figure numbers. Anything load-bearing for a
decision is restated in the text so the decision does not depend on resolving
the citation.

## What the input actually is

The catalogue entry per object is one row of the dual-mode extraction: detection
in the $r$ band, forced photometry in all other bands, 64 293 sources in total
[@bonoliMiniJPASSurveyPreview, §3.4, §4.1, Tab. 5]. The bands are 54 narrow
filters of FWHM $\sim145$ Å spaced by about 100 Å, two broader filters at the
spectral extremes ($u_\mathrm{JAVA}$ at 3497 Å and J1007 at 9316 Å), and four
SDSS-like broad bands $u,g,r,i$ [@bonoliMiniJPASSurveyPreview, §2.3, Tab. 3],
giving 60 fluxes per object and a low-resolution spectrum with
$R_\lambda\sim60$ [@bonoliMiniJPASSurveyPreview, §2.3].

Four properties of these numbers drive everything below. Every figure marked
*(measured)* is computed on the shipped export
(`data/raw/miniJpas_fluxes_errors_redshifts_full.csv`, 64 327 rows) or on the
deduplicated interim table (`data/interim/objects.parquet`, 64 293 objects x 60
bands = 3 857 580 pixels) and is tabulated in
[Measured on the shipped export](#measured-on-the-shipped-export). The code to
regenerate them is the pipeline in `jpas_da/dataset.py`.

**Fluxes are signed.** For objects near the noise level, bands other than the
detection band routinely return negative fluxes; the release states explicitly
that this is not a problem but a measurement within the noise
[@bonoliMiniJPASSurveyPreview, §3.4, fn. 14]. Any transform that requires
$f_j > 0$ is therefore disqualified. *(measured)* 21.67% of all pixels and
21.75% of the pixels with a defined error are negative.

**One per cent of the pixels has no measured error, and its flux is not
missing.** 38 916 cells (1.009%) have `FLUX_RELERR_AUTO == 0.0`, which is how
the archive flags "error not computed"; `jpas_da/dataset.py:176` turns exactly
those into `mask == 0` with `sigma = NaN`. 3 302 objects (5.14%) carry at least
one such band, 60 991 are complete. *(measured)* The flux is present for all of
them — 79.3% positive, 13.6% negative, 7.2% exactly zero — so a masked pixel is
a measurement whose uncertainty is unknown, not an absent band.

Two obligations follow, and neither is in the reference implementation below as
originally written. A masked pixel must be given **zero weight** in the
likelihood: its quoted error is 0, so any formula that divides by it produces an
infinite weight and a NaN loss for 5% of the catalogue. And it must be excluded
from the norm $r$: a masked band's flux is real but unweighted, and letting it
into $\lVert\mathbf{f}\rVert$ changes the object's brightness by whatever
fraction of the power sits in that band. *(measured)* Dropping the masked bands
changes $r$ by more than 1% for 24% of the affected objects, by a factor 1.9 for
the worst 1%, and by a factor 3.9 in the worst case.

**The errors are purely multiplicative, and they vanish with the flux.**
The export supplies `FLUX_RELERR_AUTO` per band; the absolute error is formed as
$\sigma_j = f_j \rho_j$. *(measured)* $\rho$ is signed, with
$\operatorname{sign}\rho_j = \operatorname{sign} f_j$ for all 3 820 671 defined
cells, so $\sigma_j \ge 0$ identically and
$\lvert\sigma_j\rvert = \lvert f_j\rvert\lvert\rho_j\rvert$ to machine precision
(max deviation $2\times10^{-16}$). There is **no additive term anywhere in the
quoted error**: as $f_j \to 0$, $\sigma_j \to 0$. In shape space the statistical
part becomes $v_j = \lvert g_j\rvert\lvert\rho_j\rvert$; the code adds the
multiplicative term on top, giving
$v_j=\sqrt{(g_j\rho_j)^2+(0.026g_j)^2}$, which is still exactly proportional to
$\lvert g_j\rvert$ and so vanishes wherever $g_j$ does — a band that measured no
flux claims infinite precision. 5.45% of defined
cells have $\lvert g_j\rvert < 0.01$. This single property is what makes the
additive floor of [Part II](#the-additive-floor-is-mandatory) non-optional for
this data set, and it is the one place where this note's original advice ("leave
$s_j$ at zero") was wrong.

**The aperture is `AUTO`, not `PSFCOR`.** The export is generated by
`references/queries/minijpas_flux_with_errors_full.txt`, which selects
`FLUX_AUTO` and `FLUX_RELERR_AUTO` per band from `FLambdaDualObj` and
`MAG_AUTO[rSDSS]` from `MagABDualObj`. There are no `PSFCOR` columns in this
file. The argument that the PSF-homogenised aperture is the only one carrying an
honest variance [@bonoliMiniJPASSurveyPreview, §3.3] therefore does not transfer:
what is being modelled here is an auto aperture with a purely relative error, and
the error is *derived* from the flux rather than estimated from a noise model.
That is a known weakness — it is the reason the floors below exist — but it is
the data on disk, and everything in this note is written in terms of
$f_j$ and $\sigma_j$ so none of the derivation depends on the spelling.

The aperture is still restricted, so the total-flux caveat stands in modified
form: `AUTO` is a Kron-like aperture grown in the detection band, so the fraction
of the total flux inside it depends on the light profile. That is acceptable for
an SED-shape embedding and unacceptable if the latent is later read as a
luminosity.

!!! warning "The band axis in this repository is not in one unit"
    `jpas_da.bands.wavelengths()` returns 378.0 for `J0378` (the filter's name
    encodes nanometres) but 3497.0 for `uJAVA` and 9316.0 for `J1007`
    (angstroms), all under the field name `wavelength_aa`. The *ordering* is
    correct — the sequence is monotonic — so a convolution along the axis is
    unaffected, but anything that uses the values as a physical wavelength (a
    kernel width, a redshift stretch, the `wavelength_nm` column in
    `notebooks/explore_data.py`, which divides by 10 and so plots `J0378` at
    37.8 nm) is wrong by a factor of 10 for the 54 narrow bands. Fix the unit
    before using the axis for anything but ordering.

### Measured on the shipped export

Reproduce with `make data`; the counts below are the interim table.

| Quantity | Measured |
|---|---|
| Objects / pixels | 64 293 / 3 857 580 |
| Pixels with no measured error (`relerr == 0`) | 38 916 (1.009%) |
| Objects with >= 1 masked band / complete | 3 302 (5.14%) / 60 991 |
| Pixels with negative flux | 21.67% |
| Pixels with flux exactly 0 (all masked) | 2 784 |
| Signed relative error: negative fraction | 21.75% |
| `sigma` from `flux * relerr`: negative fraction | 0% |
| $\log_{10}\lVert\mathbf{f}\rVert$, 1st to 99th percentile | 1.54 to 4.39 (2.85 dex) |
| `MAGrSDSS`, 1st / median / 99th | 17.28 / 23.59 / 25.77 |
| Objects fainter than r = 23 | 66.0% |
| Median $\lvert\rho_j\rvert$, `rSDSS` ... `J0378` | 0.216 ... 0.959 (ratio 4.4) |
| Median `v_j`, `rSDSS` ... `J0450` | 0.0168 ... 0.1526 (ratio 9.1) |
| Per-object `max v_j / min v_j` | median 13.2 |
| `sigma_l` | median 0.116, 1st/99th 0.0041/0.217 |
| Objects with `sigma_l < 0.037` | 10.8% |
| Weight `1/w^2`, multiplicative floor only: median / max / ratio | 263 / 4.9e11 / 1.85e9 (9.3 orders) |
| Fisher, multiplicative floor only: objects carrying 50% / 90% | 1 (0.002%) / 2 (0.003%) |
| Fisher, multiplicative floor only, held by `CLASS_STAR > 0.5` (10.1% of objects) | 98.8% |
| Same, with the additive floor $s_j$ at the per-band 5th percentile of $v_j$: 50%/90% | 5781 (9.0%) / 43436 (67.6%) |
| Same, star share / max weight / ratio | 0.329 / 4.9e4 / 194 |
| Effective bands (participation ratio of `1/w^2`, floors applied) | 10th/50th/90th: 15.4 / 18.8 / 23.8 |
| $\max_j g_j^2$ | median 0.130 (so $\max_j\lvert g_j\rvert$ = 0.361) |
| `half(1 + log w^2)` summed per object, floors applied | median -130.6, mean -138.8 |
| Median $\lvert\mathrm{corr}\rvert$, adjacent narrow bands, r<20 / 20-21 / 21-22 / >22 | 0.876 / 0.614 / 0.337 / 0.130 |
| `corr(uJAVA, uJPAS)`, same bins | 0.907 / 0.650 / 0.304 / 0.082 |

---

## Part I — Normalisation

### Two different freedoms, only one of which is harmless

Write the per-object data vector $\mathbf{f}=(f_1,\dots,f_{60})$ with errors
$\boldsymbol{\sigma}=(\sigma_1,\dots,\sigma_{60})$, and let $\mu_j$ be the
decoder's prediction for band $j$.

**Per-band multiplicative rescaling is a gauge symmetry.** For any positive
constants $c_j$,

$$
\bigl(f_j,\ \sigma_j,\ \mu_j\bigr) \;\longrightarrow\; \left(\frac{f_j}{c_j},\ \frac{\sigma_j}{c_j},\ \frac{\mu_j}{c_j}\right)
$$

leaves every term of a Gaussian likelihood invariant, because

$$
\frac{(f_j-\mu_j)^2}{\sigma_j^2} \;\longrightarrow\; \frac{(f_j-\mu_j)^2/c_j^2}{\sigma_j^2/c_j^2}
= \frac{(f_j-\mu_j)^2}{\sigma_j^2},
\qquad
\log\sigma_j^2 \;\longrightarrow\; \log\sigma_j^2 - 2\log c_j .
$$

The additive $-\sum_j 2\log c_j$ is constant in the parameters and vanishes from
every gradient. Rescaling is therefore free: use it to condition the numerics.

Note what "free" does and does not mean in a network. If the *inputs* are divided
by $c_j$, the decoder simply learns $\mu_j$ in the rescaled units and the symmetry
is realised by the weights themselves — no explicit correction is needed, and
dividing the decoder output as well would apply $c_j$ twice. The equality above is
a statement about the *likelihood*, which holds whichever space you write it in,
provided flux, error and prediction all sit in the same space. Mixing spaces is
the failure mode: an input divided by $c_j$ against a decoder output that is not
reweights the reconstruction loss by $c_j^{2}$ without anything in the code
looking wrong.

**Per-band mean subtraction is not a gauge symmetry.** Centring destroys the
quantity the model is meant to learn: $f_j - \langle f_j\rangle$ is not a flux
ratio, so the normalised vector no longer represents a spectral shape, and the
$\sigma_j$ carried alongside no longer describes the variable that was centred.

This rules out the two defaults people reach for first:

| Transform | Verdict |
|---|---|
| `StandardScaler` per band (centre and scale) | wrong — centring destroys colours |
| `MinMaxScaler` per band (shift and scale) | wrong — shifting destroys colours |
| `MaxAbsScaler` per band | safe but arbitrary; only if `c_j` reaches the decoder too |
| divide by a fixed per-band scale | safe — pure gauge |

### Per-object normalisation, and the decomposition it exposes

Let $c = \lVert\mathbf{f}\rVert = \bigl(\sum_j f_j^2\bigr)^{1/2}$ and define

$$
g_j = \frac{f_j}{c}, \qquad v_j = \frac{\sigma_j}{c}, \qquad \lVert\mathbf{g}\rVert = 1 .
$$

This is Eq. 1 of [@rodriguesMiniJPASSurveyQuasar2023a, §3], where the same scalar
divides both the flux and its error so that the ratio $\sigma_j/f_j$ — the
information content of the band — is untouched.

The choice $c = \lVert\mathbf{f}\rVert$ is not arbitrary. The map

$$
\mathbf{f} \;\longmapsto\; \bigl(\mathbf{g} = \mathbf{f}/\lVert\mathbf{f}\rVert,\ r = \lVert\mathbf{f}\rVert\bigr),
\qquad \mathbf{g}\in S^{59},\ r>0
$$

is the polar decomposition of $\mathbb{R}^{60}$: an orthogonal split of the data
into a *shape* on the unit sphere and a *scale* along a ray. It factorises the
measure,

$$
\mathrm{d}^{60}\mathbf{f} = r^{59}\,\mathrm{d}r\;\mathrm{d}\Sigma(\mathbf{g}),
$$

so shape and brightness are not a modelling convenience but an orthogonal
decomposition of the input space. Two things follow.

*The latent should be split too.* Brightness is physics — it carries luminosity
and distance — and it varies by four orders of magnitude between $r=17$ and
$r=23$. Fused into a single 60-dimensional input, it dominates the reconstruction
loss and consumes latent dimensions that should describe the SED. Feed the shape
through the encoder and pass $\ell = \log r$ to the decoder as a conditioning
input, alongside a separate scale head.

*Conditioning is necessary, not cosmetic.* The population mix changes with
brightness: the catalogue is dominated by stars at $r\lesssim18.5$ and by
galaxies at $r\gtrsim21$ [@jeakelMiniJPASJNEPSurveys2026a, §2.7]. The conditional
shape distribution $p(\mathbf{g}\mid \ell)$ is therefore not flat in $\ell$, and
a decoder that never sees $\ell$ is being asked to learn a mixture it cannot
resolve.

This is the same structural decision as in
[@rodriguesMiniJPASSurveyQuasar2023a, §3.1.1], which convolves the 60 normalised
fluxes and concatenates the $r$ magnitude only after the convolutional stack:
shape first, brightness added back at the end.

### What the normaliser does to the errors

Dividing by the *measured* norm propagates that measurement's error into every
component. For $g_j = f_j/c$ with $c^2 = \sum_k f_k^2$,

$$
\frac{\partial g_j}{\partial f_i} = \frac{\delta_{ij}}{c} - \frac{f_j f_i}{c^{3}},
$$

so, treating $c$ as fixed at its measured value,

$$
\operatorname{Cov}(g_j,g_l) \simeq \frac{1}{c^{2}}\left[\delta_{jl}\sigma_j^{2}
- \frac{f_j f_l}{c^{2}}\bigl(\sigma_j^{2}+\sigma_l^{2}\bigr)
+ \frac{f_j f_l}{c^{4}}\sum_i f_i^{2}\sigma_i^{2}\right].
$$

The diagonal part is $\sigma_j^2/c^2 = v_j^2$, which is what the likelihood uses.
The remaining terms are a rank-one correction whose size relative to the diagonal
is of order $g_j^2$, i.e. of order $1/60$ for an object with comparable flux in
all bands, but of order unity — meaning not negligible — for an object whose flux
concentrates in a few bands, as happens for a strong emission-line galaxy or a
very red object. It also grows at low signal-to-noise, where
$v_j$ is a large fraction of $g_j$ and the normaliser itself is poorly measured;
that is the regime at $r\gtrsim22.5$ in the narrow bands
[@bonoliMiniJPASSurveyPreview, §2.4.2].

Two remarks on the sign and the direction. Along the SED direction the
correction is negative — normalisation *shrinks* the apparent uncertainty in the
shape, because a common-mode fluctuation is partly divided out. Using the
diagonal approximation is thus conservative for the shape, which is the safe
error to make. And these moments are stated for the normalised vector only; they
are not a licence to ignore the correlation structure discussed in
[Part II](#the-likelihood).

!!! note "Background"
    The derivation of the Jacobian and the covariance above is standard
    error propagation and is not taken from any paper in the collection. The
    division by a measured scalar is the same device as the flux normalisation of
    [@rodriguesMiniJPASSurveyQuasar2023a, Eq. 1], but that paper does not
    propagate the normaliser's error.

### The scale variable has a different error budget

Since $\partial\ln c/\partial f_j = f_j/c^{2} = g_j/c$, the statistical
uncertainty of $\ell=\log c$ is

$$
\sigma_{\ell}^{2} = \sum_j\left(\frac{f_j\sigma_j}{c^{2}}\right)^{2}
= \sum_j g_j^{2}v_j^{2},
$$

with $g_j = f_j/c$ and $v_j = \sigma_j/c$ the normalised flux and error. Written
as a signal-to-noise this is the inverse square of the combined significance of
the whole vector. *(measured)* The realised values are much worse than the
$S/N\sim30$ example: the median is $\sigma_\ell = 0.116$, the 1st and 99th
percentiles are $0.0013$ and $0.214$, and only 11.0% of objects are below the
systematic bound $0.037$. The reason is that the *combined* significance is set
by the total flux, and the median object is $r=23.6$ with a per-band $S/N$ near
unity in most narrow bands. The example with $S/N\sim30$ describes the bright
tail, not the catalogue: *(measured)* roughly $r\lesssim20$, and 4.2% of the
objects with a measured `MAGrSDSS` are that bright (8.1% are brighter than 21).

The systematic term is far larger for the bright objects and negligible for the
rest. The absolute flux calibration of miniJPAS is good to at most $\sim0.04$ mag
[@bonoliMiniJPASSurveyPreview, §3.5], which in natural-log units is

$$
\sigma_{\ell}^{\mathrm{sys}} = 0.4\ln 10 \times 0.04 \approx 0.037 .
$$

So the scale is **systematics-limited for every object brighter than about
$r=21$**, and its likelihood variance needs a floor at that level. Without one,
the model fits calibration noise and the scale latent acquires a spurious
scatter that will be misread as intrinsic luminosity spread. *(measured)* The
crossover is real and sits where predicted: the median statistical $\sigma_\ell$
is 0.0028 at $r\,{=}\,17$–$19$, 0.0093 at $19$–$20$, 0.0201 at $20$–$21$, then
0.0439 at $21$–$22$, 0.0835 at $22$–$23$ and 0.130 fainter than 23. The fraction
of objects below 0.037 falls from 100% to 92% to 34% to 3% to 0% across those
bins.

For the shape, the relevant figure is the *relative* calibration. The
tile-to-tile $r$-band error is $\sigma_\mathrm{zpt}\simeq0.005$, and the
narrow-band comparison against synthetic photometry gives $0.006\pm0.028$ mag
[@bonoliMiniJPASSurveyPreview, §5.1, §5.2]. The correlated part,
$0.4\ln10\times0.006\approx0.0055$ in $\ln f$, is negligible next to the
statistical error; the scatter of $0.028$ mag, $\approx0.026$ in $\ln f$, is not.
That $0.026$ is the honest per-band floor for the shape and it exceeds the quoted
statistical error for bright objects. *(measured)* It binds on 15.8% of defined
cells, on 6.7% of bands for the median object, and on 45% for the 90th
percentile; it never binds for only 1.1% of objects. So it is not a rounding
correction to the quoted errors — it is a term comparable to them for the bright
half of the catalogue. But it does **not** bound the weight: it is multiplicative
and vanishes with $f_j$ exactly as the statistical term does, so on its own the
weight range stays at 1.85e9. See
[the additive floor](#the-additive-floor-is-mandatory) for the term that carries
that load.

### The alternatives, and why they are worse

| Normalisation | Scale invariant | Error structure | Verdict |
|---|---|---|---|
| none, with $\log r$ appended | no | clean | works; needs the network to separate scale itself |
| RMS, $c=\lVert\mathbf{f}\rVert$ | yes | noise spread over 60 bands | **recommended** |
| reference band, $c=f_r$ | yes | one band's noise tilts all 59 others | avoid |
| RMS, then per-band divide | yes | clean | **recommended**, if the decoded spectrum does not need physical units |

The reference-band ratio — dividing by the $r$-band flux, the "r-normalised
fluxes" input of the quasar ANN [@martinez-solaecheMiniJPASSurveyQuasar, §3.1] —
is attractive because $r$ has the highest signal-to-noise and $g_r\equiv1$ is
directly interpretable. Its defect is structural: a single $\sigma_r$ fluctuation
tilts the entire 60-dimensional vector, injecting one band's noise into all the
others. The RMS norm spreads that noise over the vector instead.

The last row is a refinement, not a replacement, and it carries a cost worth
being explicit about. Dividing by a fixed per-band constant $c_j$ first makes the
per-object norm $r$ a *weighted* norm, not a physical flux, so $\ell=\log r$
stops being a brightness. Either keep $\ell$ from the unweighted norm and accept
that the shape is no longer exactly unit-norm, or skip the per-band step
altogether. For this data set the second option is better: the RMS norm already
brings a bright and a faint object's shape vectors onto comparable footing,
because $\lVert\mathbf{g}\rVert=1$ by construction, and the remaining spread in
per-band scale is carried by $\mathbf{v}$, which the encoder receives.

### Reference implementation

```python
import torch

def normalise(flux, relerr, mask, floor_frac=0.026):
    """flux, relerr, mask: (N, 60) int8/float, canonical band order.

    mask_j == 0 means the archive did not quote an error for that band: the
    flux exists but is unweighted (not absent). It is excluded from the norm
    and gets weight zero in the loss, never sigma = 0.
    """
    mask = mask.to(flux.dtype)
    # absolute error from the archive's signed relative error; |.| because the
    # export stores relerr with the sign of the flux
    sigma = flux * relerr.abs()
    sigma = torch.sqrt(sigma**2 + (floor_frac * flux) ** 2)   # + relative calib.

    # the norm uses measured-error bands only, so a masked band cannot move the
    # brightness of the object it belongs to
    r = (flux**2 * mask).sum(-1, keepdim=True).clamp_min(0).sqrt()
    g, v = flux / r, sigma / r                                # ||g|| = 1 on the mask
    logr = r.log()
    # keepdim: l_mu comes back as (N, 1), and a 1-D sig_logr would broadcast
    # the scale term to (N, N) and try to allocate 16 GB at 64k objects
    sig_logr = (mask * g**2 * v**2).sum(-1, keepdim=True).sqrt()  # statistical only
    return g, v * mask, logr, sig_logr, mask
```

`||g|| = 1` holds over the unmasked bands, which is the point of the RMS norm and
also its limitation: the decoder must reproduce a vector constrained to a sphere,
so its output is best parameterised as a direction (for example by normalising
$\boldsymbol{\mu}$ before the loss, or by predicting an unconstrained vector and
letting the likelihood handle the radial residual). Empirical divergences should
be computed after normalising $\boldsymbol{\mu}$, or the loss penalises radial
deviations that the data cannot distinguish from the decoder failing to fit the
shape. With a mask the sphere is $S^{n-1}$ for $n = \sum_j \mathrm{mask}_j$ bands,
so $\mu$ must be normalised over the same $n$ bands and compared on those only.

Two details in that snippet are the corrections to the original. `flux * relerr`
is used rather than a separate absolute-error column because the export has none
— this is the only place where the purely relative error changes the code, and it
changes it by making the floor necessary rather than optional. And `v * mask`
zeroes the error of a masked band so the likelihood can set its weight to exactly
zero; leaving `v = 0` in place, which is what multiplying by the mask alone would
do, would instead give that band *infinite* weight.

If a per-band divide is added later for conditioning, it must be applied to the
input only, and `logr` must still come from the *unweighted* norm if $\ell$ is to
remain a brightness.

### Why there is no leakage to worry about here, unlike the asinh pipeline

`jpas_da/features.py` implements a different normalisation — `asinh(f/b_j)` with
`b_j = median|sigma_f|` per band — and that one **must** be fitted per fold: `b_j`
is a sample statistic, so computing it on all four tiles lets the held-out tile
into the transform (deviation ~0.04–0.07 dex per tile, measured). The RMS norm
has no such problem. It is per-object, computed from that object's own fluxes
with no sample statistic anywhere, so it cannot leak across a leave-one-tile-out
fold and needs no per-fold refitting. That is an argument for this design
independent of the shape/scale decomposition: it is the only one of the two that
is trivially correct under the split already in `data/interim/folds.parquet`.
The fixed constants `floor_frac` and `sig_sys` are calibration numbers from the
survey, not fitted here, so they do not reintroduce a cross-fold dependence.

---

## Part II — Loss function

### The likelihood

A flux measurement with a quoted 1σ error is a Gaussian statement, so the natural
generative model treats each band as an independent normal with
variance = quoted error plus a floor:

$$
p(\mathbf{g}\mid\mathbf{z},\ell)
= \prod_{j=1}^{60}\mathcal{N}\!\left(g_j;\ \mu_j(\mathbf{z},\ell),\ w_j^{2}\right),
\qquad
w_j^{2} = v_j^{2} + s_j^{2},
$$

with $v_j$ the propagated measurement error, already carrying the
*multiplicative* calibration term from `normalise`. The decoder emits
$\boldsymbol{\mu}(\mathbf{z},\ell)$; the variance is *known*, not learned.

$s_j$ is where this note's original advice needs reversing for *this* export. It
was described as a calibration fudge, normally zero. It is instead **load-bearing
and mandatory**, for a reason that is a property of the data and not of the
model:

*(measured)* the archive quotes only a relative error, and it is signed with the
flux, so the absolute error is $\sigma_j = f_j \rho_j$ with no additive term.
After the per-object division this gives $v_j = \lvert g_j\rvert \lvert\rho_j\rvert$,
which **vanishes exactly where the flux vanishes**. 5.45% of defined cells have
$\lvert g_j\rvert < 0.01$ and $\lvert g_j\rvert$ reaches $8\times10^{-8}$; the
smallest $v$ in the catalogue is $1.4\times10^{-6}$ while $\sigma_\ell$ for the
whole vector is median 0.116. So a single band that happened to measure almost no
flux announces infinite precision. *(measured, with the multiplicative 0.026
floor already applied to `v`)* $1/w_j^2$ spans **1.85e9** from its median (263)
to its maximum (4.9e11) — 9.3 orders of magnitude, set entirely by the additive
term that is absent. Without $s_j$ the gradient budget is decided by a handful of
near-zero-flux pixels: the **single largest Fisher-information object holds half
the total**, **2 objects hold 90%**, and `CLASS_STAR > 0.5` objects — 10.1% of the
catalogue, the bright ones — hold **98.8%**. Training on that would fit the
brightest stars and the zero-crossing pixels of everything else.

An additive per-band floor $s_j > 0$ bounds the weight and restores a sane
budget. *(measured)* Setting $s_j$ at the 5th percentile of $v_j$ within each band
moves the 50%/90% Fisher split from 1/2 objects to 5781/43436 (9.0%/67.6%), cuts
the maximum weight by **7.0 orders of magnitude** (4.9e11 to 4.9e4) and the
median-to-max ratio to 2.3 orders, and drops the bright-star share of the Fisher
information from 0.988 to 0.329. A uniform $s = 0.01$ is nearly equivalent
(8.1%/67.8%, star share 0.354); $s_j$ at the per-band 5th percentile of $v$ is
the version recommended here, because the per-band median $v_j$ differs by a
factor 9.1 and a uniform floor over-weights the deep bands relative to the
shallow ones.

So: **$s_j$ is an additive, per-band, object-independent floor, fixed, and
non-zero.** It is fixed for the same reason as before — a learned $s_j$ would
estimate the residual variance that the diagnostics below are meant to test —
and it is per-band because the quoted relative errors differ systematically
across the filter set. The multiplicative $0.026$ term from `normalise` is still
needed and is still separate: it is what makes the model heteroscedastic across
the catalogue, while $s_j$ is a fixed per-band floor. The decoder, the likelihood
and every diagnostic below depend on $w_j$ and not on how it was assembled.

That last clause is the central design decision of this note. Nothing else in the
architecture matters as much as the fact that each band enters the loss with its
own weight — and, given a purely relative error, that each weight is bounded.

### The additive floor is mandatory

Recorded separately because it is the one place where the shipped data contradicts
the note's earlier recommendation.

| Floor | Status | Why |
|---|---|---|
| multiplicative $0.026$ in `normalise` | required | relative calibration scatter; binds on 15.8% of defined cells, on 6.7% of bands for the median object and 45% for the 90th (measured) |
| additive $s_j$ per band | **required, not a fudge** | quoted error is purely relative, so $v_j \to 0$ like $g_j$; without it $1/w^2$ spans 1.85e9 (measured) |
| additive $s_j$ learned | no | would fit the residual variance the diagnostics test |
| none of the above | no | 9.3 orders of magnitude in $1/w^2$; one object holds half the Fisher information and 98.8% of it sits on the 10.1% `CLASS_STAR > 0.5` subsample (measured) |

### The objective

With $q(\mathbf{z}\mid\mathbf{g},\mathbf{v})=\mathcal{N}\bigl(\boldsymbol{\mu}_q,\operatorname{diag}(\boldsymbol{\sigma}_q^{2})\bigr)$,
the negative evidence lower bound per object is

$$
\mathcal{L}
= \underbrace{\mathbb{E}_{q}\left[\frac{1}{2}\sum_{j=1}^{60}
  \frac{\bigl(g_j-\mu_j\bigr)^{2}}{w_j^{2}}\right]}_{\text{weighted reconstruction}}
+ \underbrace{\frac{1}{2}\sum_{j=1}^{60}\log w_j^{2}}_{\text{noise normalisation}}
+ \underbrace{\frac{1}{2}\sum_{d=1}^{D}
  \bigl(\mu_{q,d}^{2}+\sigma_{q,d}^{2}-1-\log\sigma_{q,d}^{2}\bigr)}_{\text{KL to the prior}}
+ \underbrace{\frac{1}{2}\frac{\bigl(\ell-\hat\ell\bigr)^{2}}{\sigma_{\ell}^{2}+\sigma_{\mathrm{sys}}^{2}}}_{\text{scale head}} .
$$

Three features deserve attention.

*The weights are per band and per object.* $w_j$ differs between bands —
*(measured)* the median per-band $v_j$ runs from 0.0168 in `rSDSS` to 0.1526 in
`J0450`, a factor 9.1, and the quoted relative error from 0.216 to 0.959 — and
between objects, by the 2.85 dex of brightness above. *(measured)* Within a
single object the ratio $\max_j v_j / \min_j v_j$ has a median of 13.2. This is
what makes the model heteroscedastic: a bright galaxy's narrow-band measurement
and a faint one's are not the same kind of datum, and the loss knows it.

*The log term is what keeps the per-band budget from being set by noise alone.*
At the truth, band $j$ contributes $\tfrac12\bigl(1+\log w_j^{2}\bigr)$. That is
**not** identical across bands — it varies with $\log w_j$, which for this data
set spans several orders of magnitude. The correct statement is weaker and still
sufficient: it varies only *logarithmically* in the noise, so no band can dominate
the budget the way it does under MSE, where the contribution is $w_j^2$ and varies
with the square. *(measured, with both floors applied)* summed over the 60 bands
the term has median $-130.6$ and mean $-138.8$ per object, against $-246$ for the
flat $w = 10^{-2}$ that the earlier draft assumed. It is a large constant set by
the noise level, and it is
the only thing that penalises inflating $w_j$; with the floors fixed it is
constant in the parameters, and if they are ever learned it must be kept.

*The scale head uses a floored variance.* $\sigma_\mathrm{sys}=0.037$ from the
0.04 mag calibration bound [@bonoliMiniJPASSurveyPreview, §3.5] dominates the
statistical term for all but the faintest objects, which is the honest statement
that brightness is calibration-limited.

### Why a plain MSE loss fails here, specifically

Under $\mathcal{L}_\mathrm{MSE}=\sum_j (g_j-\mu_j)^2$, the expected cost
contributed by band $j$ at the truth is $w_j^2$. The loss budget is therefore
allocated **in proportion to the noise variance**: the noisiest bands receive the
most gradient, the deepest bands the least.

*(measured)* That is not a subtle mis-weighting on this export. With the
per-band floor applied, $\sum_j w_j^2$ puts **92.8%** of the MSE budget on the 54
narrow bands, 3.4% on `uJAVA`, 2.8% on the four SDSS-like broad bands and 1.0% on
`J1007`; `rSDSS` alone receives 0.06% and `gSDSS` 0.13%. So the faintest narrow
bands — which carry the emission lines the 145 Å filters exist to measure
[@hernan-caballeroMiniJPASSurveyPhotometric2021, §4.2] — dominate the objective,
while the broad $g,r,i$ bands that anchor the continuum contribute essentially
nothing. The decoder then learns to reproduce the noise in the worst bands and is
free to miss the continuum.

Under the Gaussian NLL the budget is set by $\log w_j^2$ instead of $w_j^2$ and
the spread collapses: *(measured)* the largest gap between the per-band mean
$\log w_j^2$ is 4.5 nats (`rSDSS` to `J0450`, a factor 9.5 in $w$), against a
factor of 1.85e9 in $w^2$. Every band therefore contributes $O(1)$ whatever its
noise, so the continuum is reproduced because it is *informative*, not because it
is loud.

### The errors must enter the encoder, not only the loss

Feed the network $\mathbf{v}$ alongside $\mathbf{g}$ — 120 channels rather than
60. Two independent reasons.

Empirically, this is exactly what [@rodriguesMiniJPASSurveyQuasar2023a, §4.1]
isolated: adding the error channel to a convolutional classifier improves it over
flux-only tree baselines, the gain is largest in the faintest magnitude bins, and
the tree baselines that ignore errors match the *no-error* CNN. The improvement
therefore comes from the uncertainty information itself, not from the
architecture.

Structurally, the reason is that the posterior cannot be calibrated without it.
If $q(\mathbf{z}\mid\mathbf{g})$ is computed from fluxes alone, it has no way to
know that a given $\mathbf{g}$ is a poor measurement, so every faint object must
map to a confident and wrong posterior. With $\mathbf{v}$ supplied the posterior
width follows the data quality automatically — and note what $\mathbf{v}$ actually
is for this export: *(measured)* $\mathbf{v}=\sqrt{(g_j\rho_j)^2+(0.026g_j)^2}$,
exactly proportional to $\lvert g_j\rvert$ — shape magnitude times relative
error — with brightness cancelling in the division. So the error channel is *not*
"brightness times per-band depth"; it is "how much flux sits in this band, times
how well it was measured". The uncertainty channel is therefore determined by the
shape and the relative errors alone, which is why a decoder conditioned on
$\mathbf{v}$ can weight bands without ever seeing the flux scale. Feed the mask as well — 180
channels ($\mathbf{g}$, $\mathbf{v}$, mask) rather than 120 — because 1.009% of
pixels have no quoted error and the model must be told which ones, or it will
learn a spurious feature from the zeroed `v`.

### Fix the floors; do not learn them

It is tempting to make $s_j$ trainable. Resist it, for a reason internal to this
project: the optimum of a learned $s_j$ is $s_j^{2}=\bigl\langle(g_j-\mu_j)^{2}\bigr\rangle-v_j^{2}$,
i.e. the model *estimates* the per-band residual variance. But that residual
variance is precisely the quantity the diagnostic below is meant to test. Learning
it makes the test an identity and destroys its value.

Both floors are therefore fixed, from the calibration budget of
[The scale variable has a different error budget](#the-scale-variable-has-a-different-error-budget).
The multiplicative one, about $0.026$ in fractional terms, is applied inside
`normalise` before the per-object division. The additive one, `s_shape` in the
code above, is **not** zero: set it at the per-band 5th percentile of $v_j$ over
the training fold. This is the reversal of the earlier draft's advice, and the
reason is the one thing in this note that the shipped data settles outright — a
purely relative error means $v_j$ vanishes with $g_j$, so the additive term is the
only thing standing between the likelihood and a weight range of $1.85\times10^9$
in which a single object carries half the Fisher information. See
[the additive floor](#the-additive-floor-is-mandatory).

Do not, however, tune $s_j$ per band against the per-band reduced $\chi^2$: that
would fit the residual variance the diagnostic is meant to measure, the same
objection as making it trainable. If the per-band reduced $\chi^2$ comes out large
in the blue after $s_j$ is set by the $v_j$ percentile rule, record it as a
calibration statement about the quoted errors. The scale head's $0.037$ is fixed
by the 0.04 mag absolute bound and is not a free parameter either.

With both fixed, the per-band reduced $\chi^2$ remains a real measurement rather
than a definition.

### Choosing $\beta$

Two mistakes are easy to make.

*Do not tune $\beta$ until the ELBO stops improving.* At the optimum the
reconstruction term equals $\tfrac12\sum_j(1+\log w_j^2)$. *(measured)* On this
export that is a **median of $-130.6$ and a mean of $-138.8$ per object**, not the
$-246$ quoted for a flat $w\sim10^{-2}$: the realised $w_j$ are larger than that
assumption, by the per-band floors above. It is a large negative constant set by
the noise level, unrelated to the $O(D)$ of the KL. The two terms are in different
units by construction, so their sum is not a scale-free goodness measure and is
not comparable across preprocessing choices.

*Do not assume the usual symptom is the danger.* Posterior collapse is the
standard worry with $\beta>1$, but here the decoder compresses 60 numbers into a
handful of latent dimensions and cannot afford to ignore $\mathbf{z}$. The
practical risk is the opposite and milder: an unannealed KL parks $\mathbf{z}$ at
the prior before the decoder has learned anything.

So: warm up $\beta$ linearly from 0 over the first quarter of training, then set
the target by the science. Values above one trade reconstruction for
disentanglement, which is the desired trade for interpretable axes (age,
metallicity, dust, star-formation history, redshift). A range of 2 to 4 is a
reasonable starting band; the operative check is whether a traverse along one
latent axis moves one physical property.

!!! note "Background"
    The $\beta$-VAE objective and the total-correlation decomposition are
    general machine-learning results (Higgins et al. 2017; Chen et al. 2018), not
    part of this collection. If a principled rather than an empirical $\beta$ is
    wanted, penalise total correlation and index-code mutual information
    separately from the dimension-wise KL: the failure mode that matters here —
    two latent axes both encoding redshift because the plain KL does not forbid
    it — is exactly what that decomposition targets.

### How many effective bands, and how many latents

For a location parameter the total Fisher information of the shape is

$$
\mathcal{I} = \sum_{j=1}^{60}\frac{1}{w_j^{2}} ,
$$

and because the per-band median $w_j$ spans a factor 9.2 *(measured: 0.0169 in
`rSDSS` to 0.1553 in `J0450`)*, the *effective* number of bands is well below 60
and varies across the catalogue.
Two different measures, because they answer different questions:

* how many bands carry usable information — i.e. per-band $S/N > 1$, equivalently
  $\lvert\rho_j\rvert < 1$ — *(measured)* median 60 at $r<20$, 58 at $20$–$21$,
  54 at $21$–$22$, 44 at $22$–$23$ and **32** fainter than 23 (10th/90th
  percentile 26/40 there). This counts bands, not information, and it is much
  closer to 60 than the earlier draft's "15 to 25 at $r\approx23$".
* the effective number that sets the Fisher information, the participation ratio
  $(\sum_j 1/w_j^2)^2 / \sum_j 1/w_j^{4}$ with both floors applied —
  *(measured)* 10th/50th/90th percentile 15.4/18.8/23.8 over the whole catalogue.
  Unlike the band count above, this **does** vary with magnitude: median 33.1 at
  $r<20$, 25.1 at $20$–$21$, 19.5 at $21$–$22$, 17.9 at $22$–$23$, 18.6 fainter.
  The floor is what caps it far below 60 even for bright objects, and the faint
  rise beyond $r=23$ is the floor taking over from the (now tiny) statistical
  errors.

Two consequences for the architecture. A latent width of 6 to 10 is the right
order: the SED is described by (age, metallicity, dust, star-formation-history
shape, redshift) plus a residual, and extra dimensions fit noise. And a 1-D
convolution along the wavelength axis is the right inductive bias, because it
shares weights between neighbouring bands, which also partially absorbs the
band-to-band correlation that the diagonal $w_j$ cannot represent.

### Where the Gaussian is wrong, and the cheap fix

The independence assumption fails for three reasons at once, and only one of them
is curable by widening the noise.

*Band-to-band correlation.* The narrow filters have FWHM 145 Å at roughly 100 Å
spacing, so adjacent bands overlap on the sky and in wavelength, and
$u_\mathrm{JAVA}$ and SDSS $u$ are near-duplicates
[@bonoliMiniJPASSurveyPreview, §2.3; @queirozMiniJPASSurveyQuasar2023a, §2.1].
The archive supplies a diagonal error only, so a Gaussian with diagonal
covariance is an approximation that must be stated.

*(measured)* The size of that approximation is strongly brightness-dependent, and
the note's earlier framing — that correlation is a mild, uniform approximation —
understates it in the bright tail and overstates it in the faint bulk. In shape
space, the median $\lvert\mathrm{corr}\rvert$ between adjacent narrow bands is
**0.876** for $r<20$, 0.614 at $20$–$21$, 0.337 at $21$–$22$ and 0.130 fainter
than 22; $\mathrm{corr}(u_\mathrm{JAVA},u_\mathrm{JPAS})$ follows the same trend,
0.907 down to 0.082. The reason is simply that the shape of a bright object is
measured, not noisy: at high $S/N$ the observed shape correlation approaches the
*physical* correlation of the SED across neighbouring filters, which is exactly
what one would want a model to learn, and only the faint end is dominated by the
correlated *noise* that the diagonal approximation ignores. Since 66% of the
catalogue is fainter than $r=23$, the diagonal approximation is defensible for
the bulk and least defensible where the fit is best constrained.

*Aperture–morphology coupling.* The aperture is `AUTO` in this export, not
`PSFCOR` (see [the aperture note](#what-the-input-actually-is)); either way it is a
restricted, profile-dependent aperture, so the fraction of the total flux inside
it depends on the light profile. That systematic is absent from $\sigma_j$ and is
correlated with FWHM and axis ratio. It will appear as latent axes correlated with
morphology, and no change of likelihood removes it.

*Occasional gross errors.* Unflagged neighbours and deblending failures survive
the flag cuts and are not Gaussian.

The first two produce *structured* residuals that a wider Gaussian cannot absorb;
the third is what a heavy tail fixes. A Student-$t$ with $\nu\approx4$ to $6$ and
scale $w_j$ costs one line:

$$
-\log p \;\propto\; \frac{\nu+1}{2}\sum_{j=1}^{60}
\log\!\left(1+\frac{\bigl(g_j-\mu_j\bigr)^{2}}{\nu\,w_j^{2}}\right).
$$

Run both and compare the per-band $\chi^2$ and the correlation of the latent with
FWHM. If the Student-$t$ changes the latent geometry, the Gaussian was being
driven by outliers rather than by the SED.

### What not to do

**Do not sentinel-code non-detections.** Both star–galaxy studies set
non-detections to a `MAG_AUTO` of 99 [@baquiMiniJPASSurveyStargalaxy, §2.2;
@jeakelMiniJPASJNEPSurveys2026a, §3.2] and one quasar network sets them to zero
[@martinez-solaecheMiniJPASSurveyQuasar, §3.1]. Both are correct for a decision
tree, which needs an ordered value to split on. Both are wrong for a generative
model: a sentinel places a fictitious spike at an arbitrary coordinate, and zero
asserts a flux that was measured at, say, $+0.3\sigma$. In the likelihood above
neither is needed — a forced-photometry non-detection is a measurement with a
large honest $w_j$, it contributes $O(1)$ to the loss, and it constrains in
exactly the proportion to its information. This is the Gaussian limit of a
censored (Tobit) likelihood, so it is a principled choice, not a convenience.

If a hard cut is wanted anyway, the thresholds used elsewhere in the collection
are $S/N<1.25$ for flagging non-detections in the mocks
[@queirozMiniJPASSurveyQuasar2023a, §3.3] and $S/N<1.8$ for the band below which
one network extrapolates rather than accepts a value
[@martinez-solaecheMiniJPASSurveyIdentification2022, §3.2]. Masking at either
level is a small refinement of the weighting; a sentinel is a different model.

**Do not log-transform.** Negative fluxes are physical
[@bonoliMiniJPASSurveyPreview, §3.4, fn. 14], and $\log f$ is undefined for them.

**Do not standardise per band.** Already argued above; it is the most common way
to destroy the colours that the embedding is supposed to capture.

**Do not treat a masked band as flux = 0 or as sigma = 0.** This export has
38 916 such cells (1.009%, 5.14% of objects). Writing `flux = 0` fabricates a
measurement at the extreme of the shape distribution; writing `sigma = 0` —
which is literally what `FLUX_RELERR_AUTO` contains — asserts infinite precision
and gives that band weight $1/w^2 = \infty$. Either produces NaN or garbage in the
loss. The treatment is the mask channel this repository already builds in
`jpas_da/dataset.py`: exclude from the norm, weight zero in the likelihood. 7.2%
of the masked cells *do* have exactly zero flux, which is why "flux = 0 means
masked" looks plausible and is wrong in the other 92.8%.

**Do not compute `sigma` as anything but `flux * relerr`.** There is no absolute
error column in this export. Every clone of the reference implementation that
expects one will silently use the wrong weight scale.

### Why no dereddening

Dereddening is omitted by decision, so the consequences belong on the record.

The correction is available in the archive — $E(B-V)$ from Bayestar17 and the
product $k_j E(B-V)$ tabulated per source [@bonoliMiniJPASSurveyPreview, §4.2.3] —
but **it is not in this export**:
`references/queries/minijpas_flux_with_errors_full.txt` selects 60 fluxes, 60
relative errors and 17 metadata fields, with no `EBV`, `EXTINCTION` or $k_j$
column anywhere. Dereddening would require re-running the TAP query, not just a
change of model. Independently, miniJPAS covers about one square degree, over
which the extinction varies very
little. The evidence is direct: a classifier given $E(B-V)$ and its error as
features found both negligible, "reflecting the fact that miniJPAS and J-NEP each
cover only about 1 deg², where Milky Way extinction varies very little"
[@jeakelMiniJPASJNEPSurveys2026a, §5.4].

Three things to keep in mind as a result.

*The shape latent still contains a small red tilt.* For a differential reddening
across the filter set of order $(k_{\max}-k_{\min})E\approx2.4E$ magnitudes,
$E=0.01$ gives about 0.024 mag, below the 0.04 mag calibration bound
[@bonoliMiniJPASSurveyPreview, §3.5]. It is a coherent, spatially varying offset —
absorbable by the latent but not identifiable from it.

*The mean reddening is unrecoverable anyway.* The calibration was tied to
Pan-STARRS $r$ and anchored to BOSS spectra through one offset per passband
[@bonoliMiniJPASSurveyPreview, §3.5], so any spatially constant $E(B-V)$ is
degenerate with those offsets by construction. Only the variation could ever be
corrected, and on one field there is almost none.

*Transfer is the cost.* If the encoder is ever applied to J-PAS proper, where
$E(B-V)$ varies across thousands of square degrees and the differential
reddening reaches a magnitude, the correction becomes mandatory — and because it
is multiplicative it would change the scale as well as the shape. Adding it later
invalidates the trained decoder rather than merely improving it. That is the price
of leaving it out now, and it is the thing to check first if the plan changes.

### Reference implementation of the objective

```python
import torch

def vae_loss(g, v, mask, logr, sig_logr, enc, dec, s_shape, sig_sys=0.037,
             beta=3.0, nu=None):
    """g, v, mask: (N, 60). g is unit-norm over mask == 1, v already carries the
    multiplicative floor and is zero where mask == 0. s_shape: (60,) additive,
    non-zero per-band floor."""
    w = torch.sqrt(v**2 + s_shape**2)
    w2 = w**2 * mask                              # masked bands: zero weight below

    z_mu, z_logvar = enc(torch.cat([g, v, mask], dim=-1)).chunk(2, dim=-1)
    z_std = torch.exp(0.5 * z_logvar)
    z = z_mu + z_std * torch.randn_like(z_std)

    mu, l_mu = dec(z, logr)                       # direction, predicted log-scale
    mu = mu * mask                                # compare on the same sphere
    mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # masked terms are dropped, not zeroed after the fact: v == 0 with mask == 0
    # would give 1/w^2 finite and (g - mu)^2 * 0 / w^2 == 0, which looks harmless
    # but still charges the log w^2 term for a band with no measurement
    resid2 = (g - mu) ** 2 / w2.clamp_min(1e-30)
    if nu is None:                                # Gaussian
        rec_cell = 0.5 * (resid2 + torch.log(w2.clamp_min(1e-30)))
    else:                                         # Student-t, heavy tail
        rec_cell = 0.5 * (nu + 1) * torch.log1p(resid2 / nu)
    rec = (rec_cell * mask).sum(-1)

    kl = 0.5 * (z_mu**2 + z_std**2 - 1.0 - z_logvar).sum(-1)
    # .squeeze(-1) because rec and kl are (N,) while l_mu is (N, 1): an
    # un-squeezed scale term broadcasts to (N, N), i.e. 16 GB at 64k objects
    scale = (0.5 * (logr - l_mu) ** 2 / (sig_logr**2 + sig_sys**2)).squeeze(-1)
    return (rec + beta * kl + scale).mean()
```

Three details in that snippet are deliberate. `s_shape` is **not** a fallback for
a mispriced band; per [the additive floor](#the-additive-floor-is-mandatory) it is
non-zero in every band, because the archive's relative error vanishes with the
flux. Set it at the per-band 5th percentile of $v_j$ from the training fold. The
mask is applied to the *reconstruction and the log term together*, so a band with
no measured error contributes exactly nothing — not an infinite weight, and not a
$O(1)$ constant either. And the `.detach()` on the normalising factor is the right
choice only if the decoder's output layer is linear-before-normalisation; if it
has a softplus or a bias that should set the radial part, drop the detach so the
gradient can. `[INFERENCE]` — that last point is implementation judgement.

Note the shape of `1/w2.clamp_min(1e-30)`: with real floors the clamp never
binds, and it is there only because a masked band has `w2 == 0` by construction.
If it ever binds on an unmasked band, the floor is misconfigured.

---

## Diagnostics

Run these after training; each maps to one decision above.

| Symptom | Cause |
|---|---|
| per-band reduced $\chi^2\gg1$ in the blue | `s_shape` too small there, or the quoted relative errors genuinely underestimate |
| per-band reduced $\chi^2\ll1$ overall | variance inflated, or the decoder is fitting noise |
| a handful of objects dominate the loss | the additive floor is missing or too small; check the Fisher-information concentration |
| NaN loss | a masked band got `v = 0` without the mask, so it has infinite weight |
| latent axis strongly correlated with $E(B-V)$ | expected given no dereddening; check its sign and spatial pattern are coherent |
| latent axis correlated with FWHM or axis ratio | aperture–morphology coupling; the "SED" embedding is partly a morphology embedding |
| latent traverse on the leading axis changes overall normalisation | the per-object scaling failed, or scale leaked into the shape latent |
| $\sigma_\ell$ fitted far below $0.037$ | the scale head's systematic floor is missing |
| latent axes pairwise correlated with $z_\mathrm{best}$ | plain KL does not forbid it; use the total-correlation decomposition |

Two external checks that need no new labels. Spearman correlation of each latent
axis against the columns this export actually ships — `zph`, `avg_zph`,
`zsp`/`e_zsp` and `angDist` — tests whether redshift is recoverable from the shape
embedding, as it must be for a $R_\lambda\sim60$ spectrum. There is no `odds` or
`z_best` column in the file: the query selects `zph`, `e_zph`, `avg_zph`, `zsp`,
`e_zsp`, `f_zsp` and the SDSS cross-match distance, nothing else. *(measured)*
`zph` is usable for 12 419 objects (19.3%) and `zsp` for 973 (1.51%), so the
photometric sample is the one with statistical power and the spectroscopic one is
a check on it. And a UMAP of $\mathbf{z}$ overlaid with the labelled subsample is
the representativeness test already used in this collection
[@jeakelMiniJPASJNEPSurveys2026a, §4.1]: if the latent space is meaningful, the
labelled objects populate it rather than clumping. With only 1168 `spCLASSsdss`
labels, overlaying `CLASS_STAR > 0.5` as a second colour is the cheap way to get
a 64k-object version of that check.

## Summary of the decisions

- Divide fluxes and errors by the same per-object scalar, $c=\lVert\mathbf{f}\rVert$;
  this is a gauge choice that leaves the likelihood invariant and exposes the
  orthogonal shape/scale split of $\mathbb{R}^{60}$.
- Per-band multiplicative rescaling is pure gauge; per-band centring or shifting
  is never safe, because it destroys the colours the embedding is meant to learn.
- Carry $\ell=\log\lVert\mathbf{f}\rVert$ as a separate scale variable and pass it
  to the decoder, with a variance floored at the 0.04 mag calibration bound.
- Use the Gaussian negative log-likelihood with per-band, per-object variance
  $w_j^2=v_j^2+s_j^2$, where $v_j$ already carries the multiplicative $0.026$
  calibration term and $s_j$ is an additive *per-band, non-zero, fixed* floor.
  $s_j$ is mandatory here because the archive's error is purely relative and
  vanishes with the flux, leaving a $1.85\times10^{9}$ dynamic range in the weight.
  Fix both rather than learning either.
- Mask, do not impute: 1.009% of pixels have no measured error. Exclude them from
  $\lVert\mathbf{f}\rVert$ and give them zero weight; never `sigma = 0`, which is
  what the archive literally stores and means "unknown".
- Feed $\mathbf{v}$ (and the mask) to the encoder as well as to the loss.
- Warm up $\beta$; choose its target on disentanglement grounds, not on the ELBO,
  and remember the reconstruction floor is about $-131$ per object here, not $-246$.
- Sentinel-code nothing: a forced-photometry non-detection is a noisy measurement.
- Accept the diagonal covariance as an approximation, state it, and mitigate it
  with convolution along the wavelength axis — noting it is worst for the bright
  objects (median adjacent-band $|\mathrm{corr}|$ 0.88 below $r=20$).
- The RMS norm has no fitted statistic, so unlike the `asinh` + `b_j` pipeline in
  `jpas_da/features.py` it needs no per-fold refitting and cannot leak across the
  leave-one-tile-out folds.
