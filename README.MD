# Seismic Regime Dynamics via Unsupervised Learning and Markov Chains

This project investigates global seismicity from the perspective of **complex systems and statistical physics**, aiming to identify emergent seismic regimes and characterize their temporal dynamics using data-driven methods.

Instead of modeling individual earthquakes, we adopt a **coarse-grained description** of seismic activity, where aggregated statistical states are extracted from time windows and treated as macrostates of the system. The temporal evolution between these macrostates is then modeled using **Markov chains**, providing a compact and interpretable description of global seismic dynamics.

---

## Motivation

Earthquake activity is a paradigmatic example of a complex, non-equilibrium system, driven by slow tectonic forcing and punctuated by intermittent energy release. While local and regional seismicity have been extensively studied, a unified **global and regime-based description** of seismic activity remains challenging.

This project addresses this gap by combining unsupervised learning with stochastic modeling to:
- identify statistically distinct seismic regimes,
- assess their physical coherence in space,
- and model their temporal evolution in a minimal and interpretable framework.

---

## Core Idea

1. Seismic events are aggregated into fixed temporal windows.
2. Each window is represented by a vector of physically meaningful statistical features.
3. These vectors define a coarse-grained state of the system.
4. Unsupervised clustering identifies distinct seismic regimes (macrostates).
5. The temporal evolution between regimes is modeled as a Markov process.

This approach yields a low-dimensional, interpretable description of global seismic dynamics.

---

## Methodology

The analysis pipeline implemented in this repository consists of the following steps:

1. **Data ingestion**  
   Global earthquake catalog obtained from the USGS database, filtered by magnitude threshold.

2. **Windowed feature extraction (temporal coarse-graining)**  
   Meaningful geophysical and statistical features (rates, magnitudes, energy, spatial dispersion, inter-event times) are computed for each temporal window.

3. **Dimensionality reduction**  
   Principal Component Analysis (PCA) is used to reduce redundancy and noise in the feature space.

4. **Unsupervised clustering**  
   K-Means clustering is applied in the reduced space to identify distinct seismic regimes.

5. **Physical interpretation and validation**  
   Clusters are interpreted using original (non-standardized) features and validated spatially using tectonic plate boundaries.

6. **Dynamic modeling**  
   The temporal evolution between regimes is modeled via Markov transition matrices, enabling a statistical description of regime persistence and transitions.

---

## Main Results

- The clustering consistently identifies **two distinct seismic regimes**:
  - a **basal / low-energy regime**, associated with lower event rates and energy release;
  - a **critical / high-energy regime**, characterized by higher seismic rates, energy accumulation, and spatial dispersion.

- The **Markov transition matrix** reveals a strongly asymmetric dynamics:
  - the high-energy regime is highly persistent and dynamically dominant;
  - the low-energy regime is metastable and typically short-lived.

- Spatial validation shows that the critical regime is associated with **globally connected tectonic activity**, particularly along major subduction zones (e.g., the circum-Pacific belt), while the basal regime reflects more localized or background seismicity.

Together, these results support a view of global seismicity as a **metastable relaxation process**, where transient quiet periods are intermittently interrupted by globally distributed energy release (more details can be found in main.ipynb).

---

## Perspectives and Future Work

The framework developed here is intentionally generic and extensible. Natural extensions include:

- application to climate and extreme weather events,

- identification of regime shifts in atmospheric or oceanic systems,

- coupling with early-warning indicators for critical transitions,

- Further development of a two-level system model describing seismic phenomena;

These directions make the approach particularly suitable for interdisciplinary research at the interface of geophysics, climate science, and statistical physics.

---

## Reproducibility

conda env create -f environment.yml
conda activate seismic-regime-dynamics

---
## Repository Structure

```text
seismic-regime-dynamics/
├── data/         # Raw and processed seismic catalogs
├── notebooks/    # Main analysis notebook (main.ipynb)
├── reports/      # Generated figures, maps, and diagnostics
├── src/          # Feature extraction, clustering, PCA, spatial validation
├── environment.yml
└── README.md
---











