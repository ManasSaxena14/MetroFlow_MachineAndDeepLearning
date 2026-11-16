# Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models

[Live dashboard](https://trafficflowamldl.streamlit.app) · [Research paper, IEEE format (PDF)](docs/paper/paper.pdf) · [AML report (PDF, 22 pages)](docs/report/report.pdf) · [AML slides (13)](docs/presentation/AML_Traffic_Flow_Forecasting.pptx) · [DL slides (16)](docs/presentation/DL_Graph_Models.pptx)

**Does a model that sees the road network forecast traffic better than strong per-sensor models, and where does the gain come from?**
This project forecasts freeway speed 5 to 60 minutes ahead on METR-LA (207 Los Angeles sensors). It builds a leak-free preprocessing pipeline, six forecasting baselines, and four experiments that measure what independent per-sensor models miss. It then trains two graph neural networks on the same windows, a spatio-temporal graph attention network (STGAT) and a diffusion convolutional recurrent network (DCRNN), plus a hybrid that corrects DCRNN with a small gradient-boosted tree model, and compares everything under one protocol with day-level significance tests. Signal control is simulation only; nothing here controls real traffic.

![A jam reaches the downstream sensor first, then travels back upstream](docs/figures/fig05d_propagation_example.png)

*Congestion propagates backwards along the freeway at about 16 km/h: the downstream sensor slows first. A model that sees only its own sensor's history cannot see this coming.*

An [interactive map of the sensor graph](docs/figures/fig04_sensor_graph_map.html) is also included (open it in a browser).

## Results

Test block (the last 24 days, never used for tuning); masked MAE in mph, lower is better. Full tables: [results/baselines.csv](results/baselines.csv), [results/gap_results.csv](results/gap_results.csv).

| Model | 15 min | 30 min | 60 min |
|---|---|---|---|
| Persistence | 3.49 | 4.20 | 5.37 |
| Historical average | 4.42 | 4.42 | 4.42 |
| Ridge per sensor | 3.31 | 4.08 | 5.17 |
| Random Forest (multi-output) | 3.08 | 3.68 | 4.45 |
| LSTM, no graph | 2.97 | 3.55 | 4.35 |
| **XGBoost per horizon** (best independent model) | 3.02 | 3.51 | 4.01 |
| XGBoost + neighbour speeds (both directions) | **2.91** | **3.42** | **3.95** |

**What the experiments show**
- XGBoost is the strongest independent model; its error grows steadily with the forecast horizon.
- In the 30 minutes after a congestion onset its error is 5.9 to 6.4 times larger than at other times (3,911 onsets in the test period).
- The errors of road-linked sensors are correlated (0.20 at 60 minutes for sensors one link apart) while those of unconnected sensors are not.
- Adding hand-made neighbour speeds lowers error at every horizon (up to 3.5%), so the missing information is spatial, but one-hop features capture only part of it. That is the case for a graph model.
- Weekday rush-hour error is 1.4 to 1.95 times off-peak error for every model.

### Deep learning (Phase 2): graph models and the hybrid

Same test block, windows, mask and metric code as the baselines. Graph models and the hybrid: mean of 3 seeds. Full tables: [results/final.csv](results/final.csv), [results/significance.csv](results/significance.csv), [results/ablations.csv](results/ablations.csv), [results/hybrid_ablation.csv](results/hybrid_ablation.csv).

| Model | 15 min | 30 min | 60 min | RMSE 60 | Peak MAE 60 |
|---|---|---|---|---|---|
| XGBoost per horizon (best per-sensor model) | 3.02 | 3.51 | 4.01 | 7.28 | 6.22 |
| STGAT, graph attention (ours) | 2.84 | 3.32 | 3.95 | 7.96 | 6.07 |
| DCRNN, diffusion graph model | 2.64 | 3.05 | 3.57 | 7.67 | 5.40 |
| **Hybrid: DCRNN + XGBoost residual corrector (ours)** | **2.62** | **3.01** | **3.47** | 7.30 | **5.20** |

- **DCRNN beats XGBoost by 12.3% / 13.1% / 11.0%** at 15 / 30 / 60 minutes, on all 24 test days (paired bootstrap over days and Diebold-Mariano test, p < 0.0001). STGAT also beats XGBoost, by 5.9% at 15 minutes but only 1.3% at 60.
- **The hybrid** keeps DCRNN as the backbone and lets a shallow, regularised XGBoost model learn DCRNN's residual from time of day, recent change and both forecasts, fitted on the validation block only. It lowers DCRNN's error by a further 0.8% / 1.2% / 2.6% (better on at least 23 of 24 days, p < 0.0001). A corrector on all 46 features overfits; see `results/hybrid_ablation.csv`.
- **The gain comes from the road graph.** Without a graph STGAT falls back to the no-graph LSTM (4.35 at 60 minutes); a random graph with as many links is worse by 0.22; the gain of DCRNN over XGBoost grows from 4.3% for sensors with 0-4 links to 12.8% for sensors with 13 or more.
- **Listening downstream matters.** Jams reach the downstream sensor first, so STGAT receives messages over both directions of each link: downstream-only beats upstream-only and both is best; DCRNN without its backward random walk loses 0.15 at 60 minutes. The trained attention moves downstream (52% to 62%) when a slowdown is 30 minutes away, for 149 of 163 sensors.
- **Our DCRNN port reproduces the published numbers.** Under the reference protocol it reaches 2.63 / 3.03 / 3.54 mph against the published 2.67 / 3.08 / 3.56 (within 1.5%); the stricter protocol used for all other results costs about 0.5%.
- **Right after congestion starts** DCRNN's 15-minute error is 13.3 versus 16.2 mph for XGBoost. RMSE shows the flip side: DCRNN's rarer misses are larger, and the hybrid brings its 60-minute RMSE from 7.67 to 7.30.

![Test error by forecast horizon](docs/figures/fig15_final_error_by_horizon.png)

### Model accuracy

Speed forecasting is a regression task, so there is no single accuracy figure. Two plain-language scores on the test block ([results/accuracy.csv](results/accuracy.csv)): **percent accuracy** (100 minus MAPE) and the share of forecasts **within 5 mph** of the true speed.

| Model | Percent accuracy (15 / 30 / 60 min) | Within 5 mph, % (15 / 30 / 60 min) |
|---|---|---|
| Persistence | 91.4 / 89.2 / 85.5 | 80.8 / 77.9 / 73.5 |
| Historical average | 86.1 / 86.1 / 86.1 | 76.5 / 76.5 / 76.5 |
| Ridge per sensor | 91.1 / 88.3 / 84.5 | 82.3 / 77.9 / 70.6 |
| Random Forest | 91.7 / 89.4 / 86.5 | 84.1 / 80.6 / 75.9 |
| LSTM (no graph) | 92.0 / 89.7 / 86.5 | 85.1 / 82.6 / 79.5 |
| **XGBoost (best independent model)** | 91.8 / 89.9 / 88.0 | 84.3 / 81.4 / 78.6 |
| **XGBoost + neighbours (both directions)** | 92.3 / 90.4 / 88.3 | 84.7 / 81.7 / 78.8 |

These percentages look high mainly because traffic flows freely most of the time, so the error in mph and the error during congestion are the more informative quantities. See the report for RMSE, MAPE, peak versus off-peak and validation scores.

## Research foundation

The project follows the hybrid logic of the brief (segment-level machine learning first, network-aware deep learning second) and builds on published work; the report maps each reference to its role in a literature table and the slides summarise it.

| Work | Idea | Used here for |
|---|---|---|
| Li et al., ICLR 2018 (DCRNN) | traffic as diffusion on a directed graph with bidirectional random walks; 12 to 15% gain over its baselines | data, graph and protocol reference; model to reproduce next; motivates the upstream/downstream experiment |
| Veličković et al., ICLR 2018; Brody et al., ICLR 2022 | graph attention over neighbours; GATv2 dynamic attention | STGAT, our graph attention model |
| Yu et al., IJCAI 2018; Wu et al., IJCAI 2019 | graph forecasting with convolutional structures; learned adjacency | related work; learned adjacency is a listed extension |
| Wolpert 1992; Diebold and Mariano 1995 | stacked generalisation; comparing forecast accuracy | the hybrid's design; day-level significance tests |
| Chen and Guestrin 2016; Breiman 2001; Hochreiter and Schmidhuber 1997; Hoerl and Kennard 1970 | XGBoost, Random Forest, LSTM, Ridge | the four learned baselines |
| Kaufman et al. 2012; Hyndman and Koehler 2006 | data leakage; properties of accuracy measures | purge gap and leakage tests; how MAE, RMSE and MAPE are read |
| Zhang et al., WWW 2019 (CityFlow) | reinforcement-learning signal simulator | optional, simulation-only signal control |

Four hypotheses derived from this literature were tested and all were supported: (H1) error grows with the horizon, (H2) errors concentrate after congestion onsets, (H3) residuals correlate along road links, (H4) neighbour information helps, downstream more than upstream, but one-hop features do not close the gap (report, Section 8.1).

## Method in brief

- **Leak-free protocol.** Timestamps are split 70/10/20 into train, validation and test; windows whose targets straddle a boundary are purged, so no target is in two splits. The scaler is fitted on training data only, and a test proves that the reference window-level split lets train and validation targets overlap.
- **Outages are masked, not treated as zero traffic.** Missing readings are NaN targets with an explicit mask; every loss and metric ignores them.
- **Equal history for every model.** The last 12 steps (60 minutes) in, the next 12 steps (5 to 60 minutes) out.
- **Six baselines, one evaluation path.** Persistence, historical average, Ridge per sensor, XGBoost per horizon, a multi-output Random Forest, and an LSTM with shared weights and no graph. Hyper-parameters are chosen on validation only.
- **Gap experiments.** Error against horizon, error after congestion onsets, residual correlation along road links, and a neighbour-information test.
- **Graph models (Phase 5).** STGAT: a shared GRU per sensor, then two GATv2 attention layers over both directions of every road link (edge features: link weight, direction). DCRNN: a PyTorch port of the reference diffusion model. Masked MAE in mph, early stopping on validation, 3 seeds, 8 ablations. The attention layer is tested to match PyTorch Geometric's `GATv2Conv`.
- **Hybrid.** DCRNN plus a per-horizon XGBoost residual corrector (depth 3, at least 200 samples per leaf) fitted on the validation block and scored once on test.
- **Fair comparison (Phase 6) and maps (Phase 7).** One table for every model, paired day-level significance tests, gains by connectivity and after congestion onsets, congestion maps, an error map and a time-space diagram.

## Repository structure

```
.
├── app.py                  Streamlit dashboard over the saved results (deployed)
├── Makefile                one command per stage (see below)
├── configs/data/           split, window, masking and peak definitions
├── src/
│   ├── data/               loading, leak-free preprocessing, readers
│   ├── features/           46-column per-sensor feature table
│   ├── models/             baselines, XGBoost and Random Forest, LSTM, graph layers, STGAT, DCRNN, hybrid
│   ├── eval/               masked metrics, per-horizon and peak evaluation, accuracy
│   ├── analysis/           exploratory analysis and gap analysis
│   ├── train/              runners for the baselines, gap experiments, graph models and the hybrid
│   └── viz/                figures
├── tests/                  automated tests (data, leakage, baselines, gap, graph models, comparison, hybrid)
├── results/                every result table (CSV and JSON)
├── notebooks/              exploratory data analysis
├── scripts/                data download, graph-model runs, report / paper table and slide generators
└── docs/
    ├── paper/              IEEE-format research paper (LaTeX source, generated tables, PDF)
    ├── report/             LaTeX report of the AML part, bibliography, compiled PDF
    ├── presentation/       slide decks (AML part, DL part)
    └── figures/            figures used by the report and README
```

## Getting started

```bash
make setup        # Python 3.11 environment (uv) and requirements-pipeline.txt; macOS: brew install libomp
make data         # METR-LA and the sensor graph (graph files from the DCRNN repository)
make eda          # exploratory analysis: figures and results/phase1_findings.json
make preprocess   # leak-free windows -> data/processed/strict/
make features     # per-sensor feature tables
make baselines    # six baselines -> results/baselines*.csv
make gap          # gap experiments -> results/gap*/
make accuracy     # accuracy summaries -> results/accuracy.csv
make graph        # graph models: STGAT and DCRNN x 3 seeds, ablations (~10 h on an Apple M3 Pro, resumable)
make hybrid       # DCRNN + XGBoost residual corrector -> results/hybrid*
make phase5       # ablation table, learning curves, attention inspection
make compare      # final comparison and significance tests -> results/final*.csv, results/significance.csv
make maps         # congestion maps, error map, time-space diagram
make slides       # DL slide deck from results/
make paper        # IEEE paper from results/ -> docs/paper/paper.pdf (brew install tectonic)
make report       # regenerate tables from results/ and compile docs/report/report.pdf (brew install tectonic)
make test lint
make dashboard    # open the dashboard locally
```

The deployed dashboard at [trafficflowamldl.streamlit.app](https://trafficflowamldl.streamlit.app) reads only the pre-computed files in `results/` and installs only `requirements.txt` (the full pipeline environment is `requirements-pipeline.txt`). Intermediate data (about 2.5 GB per protocol) is not committed.

## Documentation

- **[Project report](docs/report/report.pdf)**: 22 pages with abstract, literature review, data, methodology, all results, the gap analysis, software and deployment, limitations and a bibliography. The LaTeX source is in `docs/report/`; every table and quoted number is generated from `results/` by `scripts/make_report_tables.py`, and `make report` rebuilds it.
- **[Research paper](docs/paper/paper.pdf)** (IEEE conference format): the whole project, with the deep-learning literature review, the graph models, the hybrid, ablations, significance tests and congestion maps. Every number is generated from `results/` by `scripts/make_paper_tables.py`; `make paper` rebuilds it.
- **[DL slides](docs/presentation/DL_Graph_Models.pptx)**: 16 slides with speaker notes on the graph models, the hybrid, the comparison, ablations, attention and maps; generated from `results/` by `scripts/make_dl_slides.py`.
- **[AML slides](docs/presentation/AML_Traffic_Flow_Forecasting.pptx)**: 13 slides covering the problem, research foundation, data, leak-free pipeline, baselines, all scores, the gap, conclusion and references.
- **[Live dashboard](https://trafficflowamldl.streamlit.app)**: interactive view of the saved results (`app.py`, contributed by Sumit).

## Data

METR-LA ([Jagadish et al., 2014](https://doi.org/10.1145/2611567)) as distributed with the [DCRNN](https://github.com/liyaguang/DCRNN#data-preparation) code: 34,272 five-minute speed readings from 207 sensors, 1 March to 27 June 2012, with a directed road-distance adjacency matrix. 8.1% of readings are sensor outages (0 mph), which are masked.

## Limitations

METR-LA records speed only, so flow and occupancy features need PeMS data. The data are 2012 freeway readings from one city. The per-sensor baselines were trained once (seed 42); graph models and the hybrid use 3 seeds, ablations one seed. The hybrid is fitted on the validation block, which the baselines did not use for fitting. Signal control was not run: METR-LA freeways have no signals, so it would have to be a simulation of a synthetic network.

## References

The full bibliography, with DOIs and arXiv identifiers, is in [docs/report/references.bib](docs/report/references.bib).

1. Y. Li, R. Yu, C. Shahabi, Y. Liu. *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* ICLR 2018. [arXiv:1707.01926](https://arxiv.org/abs/1707.01926)
2. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. *Graph Attention Networks.* ICLR 2018. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
3. B. Yu, H. Yin, Z. Zhu. *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.* IJCAI 2018. [arXiv:1709.04875](https://arxiv.org/abs/1709.04875)
4. Z. Wu, S. Pan, G. Long, J. Jiang, C. Zhang. *Graph WaveNet for Deep Spatial-Temporal Graph Modeling.* IJCAI 2019. [arXiv:1906.00121](https://arxiv.org/abs/1906.00121)
5. H. Zhang et al. *CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenario.* WWW 2019. [arXiv:1905.05217](https://arxiv.org/abs/1905.05217)
6. T. Chen, C. Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
7. L. Breiman. *Random Forests.* Machine Learning 45(1):5-32, 2001.
8. S. Hochreiter, J. Schmidhuber. *Long Short-Term Memory.* Neural Computation 9(8):1735-1780, 1997.
9. A. E. Hoerl, R. W. Kennard. *Ridge Regression: Biased Estimation for Nonorthogonal Problems.* Technometrics 12(1):55-67, 1970.
10. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
11. H. V. Jagadish et al. *Big Data and Its Technical Challenges.* Communications of the ACM 57(7):86-94, 2014. [doi:10.1145/2611567](https://doi.org/10.1145/2611567)
12. S. Kaufman, S. Rosset, C. Perlich, O. Stitelman. *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD 6(4), 2012.
13. R. J. Hyndman, A. B. Koehler. *Another Look at Measures of Forecast Accuracy.* International Journal of Forecasting 22(4):679-688, 2006.

## Contributors

Team T036, Newton School of Technology, Rishihood University. The work is divided equally: each member owns one part of
the machine-learning phase and one part of the deep-learning phase.

| Member | Phase 1: machine learning | Phase 2: deep learning |
|---|---|---|
| Harsh Raj (2401010180) | project setup, data loading, exploratory analysis and the congestion-propagation finding | STGAT graph attention model, graph layers and attention inspection |
| Manas Saxena (2401010260) | leak-free preprocessing, the six baselines, AML report and slides | fair comparison and day-level significance tests |
| Adil Mirza (2401020005) | feature engineering, XGBoost and Random Forest tuning | DCRNN model, ablation studies and the published-number reproduction |
| Augustya Purohit (2401010114) | gap analysis: horizon, onset, residual and neighbour experiments | hybrid DCRNN + XGBoost model and the IEEE research paper |
| Sumit Yadav (2401010464) | Streamlit dashboard and accuracy metrics | congestion maps, time-space diagram and the DL slides |
