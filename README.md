# ASZED: Mechanistic Etiology via Hopf Bifurcation

This repository contains the computational pipeline and generative models establishing a mechanistic etiology for schizophrenia using EEG data (ASZED dataset). 

By modeling the brain's macroscopic dynamics through a Supercritical Hopf Bifurcation, we demonstrate that the clinical condition emerges as a topological state-space collapse (from a healthy limit cycle to a noise-dominated point attractor), rather than a simple amplitude reduction.

## Core Findings
* **1/f Hardware Control:** The pipeline strictly isolates true phase dynamics from the 1/f aperiodic background.
* **Topological Collapse:** Patients exhibit a profound reduction in state-space radius and path length.
* **Complexity Paradox:** Despite the volumetric collapse, effective dimensionality ($d_{eff}$) increases in patients, indicating fragmentation and trapping in local thermal noise.

## Repository Structure
* `FINAL_1F_FALSIFICATION_PIPELINE.py`: Destroys the 1/f spectrum to prove the etiology lies in the nonlinear dynamics.
* `TOY_ASZED_HOPF_BIFURCATION.py`: The pure theoretical physics model generating the pathology from differential equations.
* `MASTER_EDF_FULL_PIPELINE.py`: The subject-level empirical feature extraction pipeline.

## Installation
```bash
git clone [https://github.com/Dasein193/ASZED-Mechanistic-Etiology.git](https://github.com/Dasein193/ASZED-Mechanistic-Etiology.git)
cd ASZED-Mechanistic-Etiology
pip install -r requirements.txt

## Scientific Significance: The Hopf Transition as Etiology

Historically, Schizophrenia has been described through phenomenological observations or purely statistical markers (e.g., power spectral density changes). This project moves beyond "what" is happening to "why" it happens.

By implementing a **Supercritical Hopf Bifurcation** as the generative engine, we provide a mechanistic explanation for the ASZED dataset findings:

1.  **Healthy State (Supercritical):** The system operates in a stable Limit Cycle. High coherence, expansive state-space exploration, and a clear distinction between signal and background noise.
2.  **Pathological State (Subcritical Collapse):** As the bifurcation parameter ($\mu$) crosses the critical threshold, the limit cycle vanishes. The system collapses into a Point Attractor. 
3.  **The Result:** The emergent "signal" is no longer a robust oscillation but a trajectory trapped in local thermal noise. This explains why we observe a volumetric collapse in state-space radius alongside an increase in effective dimensionality ($d_{eff}$)—the complexity of the pathological state is not high-order coordination, but fragmented stochasticity.

This model proves that the etiology of the condition is a **topological phase transition** of the brain's global neurodynamics.
