# Hybrid Fuzzy Inference Model with a Dynamic Threat Coefficient for National Cybersecurity State Assessment

Code and data for the paper:

> A. Rakhysh, M. Kunelbayev, S. Adilzhanova, Z. Primzhanov, B. Baimyrza, "Hybrid Fuzzy Inference Model with a Dynamic Threat Coefficient for National Cybersecurity State Assessment: A Case Study of Kazakhstan," *International Journal of Computer Network and Information Security (IJCNIS)*, 2026 (under review).

The model combines:

- Fishburn-weighted aggregation of NCSI 3.0 indicators into a baseline protection level **CS**;
- a threat intensity parameter **Y1**, the equal-weight mean of four components:
  - C1 = 1 − NRI;
  - C2 = weakness of protective mechanisms;
  - C3 = weakness of incident and crisis management;
  - C4 = incident intensity from KZ-CERT statistics;
- a Mamdani fuzzy inference system with 3 input terms and 9 rules. Implication is min, aggregation is max, and defuzzification is the discrete centroid with Δz = 0.001;
- a dynamic threat change coefficient **D(t) = Y1(t) − Y1(t−1)** and a stability index **CSI = CS_final · (1 − |D(t)|)**.

## Repository structure

| File | Content |
| --- | --- |
| `fis.py` | The model: Fishburn weights, membership functions, rule base, Mamdani and Sugeno inference, Y1 components, D(t), CSI |
| `data.py` | NCSI 3.0 group points, NCSI and NRI scores of all observations, KZ-CERT incident counts |
| `data/kzcert_incidents_2021_2026.csv` | KZ-CERT incidents by type, 2021–2026 (Table 10 of the paper) |
| `experiments.py` | All computations, written to `results.json` and `surface.npy` (see below) |
| `figures.py` | Figs. 3–10 |
| `figures_schemes.py` | Figs. 1–2 (model structure, Mamdani system) |
| `results.json` | Reference output of `experiments.py` |
| `fig/` | Generated figures (300 dpi) |

`experiments.py` covers:

- base case (Section 3.2);
- multi-country comparison with baselines and ablation (Tables 7–8);
- dynamic evaluation (Table 6);
- sensitivity analysis (Table 9);
- rule-base checks and run time (Section 3.7).

## Requirements

Python ≥ 3.10 with NumPy, SciPy and Matplotlib. The results in the paper were obtained with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
python experiments.py        # -> results.json, surface.npy (about 10 s)
python figures.py            # -> fig/fig3 ... fig10
python figures_schemes.py    # -> fig/fig1, fig/fig2
```

All computations are deterministic except the Monte Carlo sensitivity analysis, which uses a fixed random seed (42). A fresh run therefore reproduces `results.json` exactly. The only difference is the measured run time.

## Data sources

- **NCSI 3.0.** National Cyber Security Index, e-Governance Academy, https://ncsi.ega.ee. Group points were taken from the country pages; the versions used are listed in `data.py` and Table 11 of the paper.
- **NRI.** Network Readiness Index, Portulans Institute, https://networkreadinessindex.org.
- **KZ-CERT.** Statistics of information-security incidents registered in Kazakhstan, https://cert.gov.kz. Data for 2026 cover 1 January – 24 November 2026 and are provisional.

## License

Released under the MIT License (see `LICENSE`). The NCSI, NRI and KZ-CERT data remain the property of their publishers.

## Citation

If you use this code, please cite the paper above. Volume, issue, pages and DOI will be added after publication.
