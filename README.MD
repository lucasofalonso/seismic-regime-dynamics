# Seismic Regime Dynamics via Unsupervised Learning and Markov Chains

This project investigates seismicity from the perspective of **complex systems and statistical physics**, aiming to identify emergent seismic regimes and characterize their temporal dynamics using data-driven methods.

Instead of modeling individual earthquakes, we adopt a **coarse-grained description** of seismic activity, where aggregated statistical states are extracted from time windows and treated as macrostates of the system. The temporal evolution between these macrostates is then modeled using **Markov chains**, providing a compact and interpretable description of seismic dynamics.

> **Status:** Active development. The pipeline is functional and produces physically meaningful results. Current focus is on regional validation and formal statistical testing of the Markov hypothesis.

---

## Motivation

Earthquake activity is a paradigmatic example of a complex, non-equilibrium system driven by slow tectonic forcing and punctuated by intermittent energy release. This project addresses the challenge of a **regime-based description** of seismic activity by combining unsupervised learning with stochastic modeling to:

- identify statistically distinct seismic regimes,
- assess their physical coherence in space and time,
- and model their temporal evolution in a minimal, interpretable framework.

---

## Core Idea

1. Seismic events are aggregated into sliding temporal windows.
2. Each window is represented by a vector of physically meaningful statistical features.
3. These vectors define a coarse-grained state of the system.
4. Unsupervised clustering identifies distinct seismic regimes (macrostates).
5. The temporal evolution between regimes is modeled as a Markov process.

---

## Methodology

1. **Data ingestion** — Global earthquake catalog from the USGS API, filtered by magnitude threshold and optionally by geographic region.

2. **Data storage** — Raw catalog ingested into a local SQLite database, enabling structured queries and regional filtering via SQL before loading into memory.

3. **Windowed feature extraction** — Geophysical and statistical features (event rates, magnitudes, energy, spatial dispersion, inter-event times) computed per temporal window.

4. **Dimensionality reduction** — PCA applied to reduce redundancy and noise in the feature space.

5. **Unsupervised clustering** — K-Means in the reduced space to identify distinct seismic regimes.

6. **Physical interpretation** — Clusters characterized using original (non-standardized) features and validated spatially against tectonic plate boundaries.

7. **Markov modeling** — Transition matrices estimated from the label sequence, with formal statistical testing of the memoryless property via dwell time analysis and KS tests against the exponential distribution.

8. **Supervised validation** — CNN-1D trained on temporal sequences of features to assess the predictability of regime transitions.

---

## Main Results

**Global analysis** (`main.ipynb`) identifies two statistically distinct seismic regimes — a basal low-energy regime and a critical high-energy regime — with a strongly asymmetric Markov dynamics favoring persistence in the active state. Spatial validation links the critical regime to globally distributed tectonic activity along major subduction zones.

**Regional analysis** (`sensitivity_analysis.ipynb`) focuses on the **Indonesia/Sumatra subduction zone** and produces the most robust results to date:

- Two physically distinct regimes emerge consistently across all window configurations, **including those without overlap** (silhouette > 0.33), ruling out autocorrelation artifacts as the primary driver of cluster separation.
- The **basal regime** (~4 events/day, spatially localized, ~55 km dispersion) and the **active regime** (~11 events/day, spatially dispersed, ~1000 km along the subduction zone) are stable across all tested timescales.
- Theoretical and empirical dwell times agree closely (`τ_basal ≈ 40h`, `τ_active ≈ 10h`), supporting the Markov approximation as a valid first-order description.
- Formal KS tests reveal **heavy-tailed dwell time distributions**, consistent with long-range temporal correlations and potentially with **self-organized criticality (SOC)** — motivating extensions beyond first-order Markov models.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `main.ipynb` | Full exploratory pipeline: data ingestion, feature extraction, PCA, clustering, Markov modeling, spatial validation |
| `sensitivity_analysis.ipynb` | Robustness analysis over window configurations; regional focus on Indonesia/Sumatra; dwell time analysis and memoryless test |

---

## Perspectives and Future Work

- **Beyond first-order Markov**: the heavy-tailed dwell time distributions suggest memory effects consistent with SOC. Higher-order chains or renewal processes with power-law waiting times are natural extensions.
- **Regional catalog expansion**: Japan/Kuril, Nazca/Andes, and Alaska subduction zones as natural test cases for cross-regional comparison.
- **Two-level system formalization**: treating the binary regime dynamics within the TLS framework of statistical physics, including barrier estimation and relaxation time analysis.
- **Generalization**: the pipeline is domain-agnostic. Direct extensions include volatility regime detection in financial time series and extreme event identification in climate data.

---

## Reproducibility
```bash
conda env create -f environment.yml
conda activate seismic-regime-dynamics
```
---

### Data

Raw seismic data is not versioned. To reproduce:

1. Run the ingestion cells in `notebooks/exploratory_analysis.ipynb` to fetch data from the USGS API, or place your own catalog at `data/raw/usgs_3y_minmag4p5.parquet`
2. Create the SQLite database:
```bash
python src/data_ingestion/to_sqlite.py
```
This populates `data/seismic.db` with indexed tables used by the analysis notebooks.


---

## Repository Structure
```text
seismic-regime-dynamics/
├── data/                  # Seismic catalogs and SQLite database (not versioned, see Reproducibility)
├── figures/               # Plots of silhouette, Markov matrices and dwell times (Indonesia)
├── notebooks/             # exploratory_analysis.ipynb, sensitivity_analysis.ipynb
├── reports/               # Interactive map and diagnostics
├── src/                   # Modular pipeline (ingestion, SQLite, features, PCA, clustering, spatial)
├── environment.yml
└── README.md
```