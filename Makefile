PY := .venv/bin/python

.PHONY: setup data eda preprocess features baselines gap accuracy graph hybrid phase5 compare maps slides paper report dashboard test lint all

setup:            ## create the environment (Python 3.11 via uv) and install pinned requirements
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements-pipeline.txt
	@echo "macOS: XGBoost also needs 'brew install libomp'; the report needs 'brew install tectonic'"

data:             ## download METR-LA and the sensor graph into data/raw
	./scripts/download_data.sh

eda:              ## execute the EDA notebook: figures 1-5 and results/phase1_findings.json
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb

preprocess:       ## leak-free windows -> data/processed/<protocol>/
	$(PY) -m src.data.preprocess

features:         ## per-sensor feature tables (memory-mapped .npy)
	$(PY) -m src.features.build

baselines:        ## six baselines -> results/baselines*.csv and figures 6-7
	$(PY) -m src.train.baselines persistence hist_avg ridge xgb rf lstm --peak
	$(PY) -m src.viz.baseline_plots

gap:              ## gap experiments -> results/gap_results.csv, results/gap/ and figures 8-11
	$(PY) -m src.train.phase4 neighbours onset residuals table
	$(PY) -m src.viz.gap_plots

accuracy:         ## percent accuracy and within-5-mph scores -> results/accuracy.csv
	$(PY) -m src.eval.accuracy

graph:            ## Phase 5: STGAT and DCRNN (3 seeds), ablations, reference-protocol DCRNN (~10 h on an M3 Pro; resumable)
	./scripts/run_phase5.sh all

hybrid:           ## Phase 5 hybrid: DCRNN + regularised XGBoost residual corrector (and blend / full ablations)
	$(PY) -m src.train.hybrid

phase5:           ## Phase 5 analysis: ablation table, learning curves, attention inspection -> results/, figures 12-14
	$(PY) -m src.eval.phase5

compare:          ## Phase 6: final fair comparison, significance tests, gain breakdown -> results/final*.csv, figures 15-17
	$(PY) -m src.eval.compare

maps:             ## Phase 7: congestion maps, error map, time-space diagram, animated HTML -> figures 18-21
	$(PY) -m src.viz.maps

slides:           ## regenerate the DL-part slide deck from results/ (docs/presentation/DL_Graph_Models.pptx)
	$(PY) scripts/make_dl_slides.py

paper:            ## IEEE-format research paper from results/ -> docs/paper/paper.pdf (brew install tectonic)
	$(PY) scripts/make_paper_tables.py
	cd docs/paper && tectonic paper.tex

report:           ## regenerate the report tables from results/ and compile docs/report/report.pdf
	$(PY) scripts/make_report_tables.py
	cd docs/report && tectonic report.tex

dashboard:        ## interactive Streamlit dashboard over results/ (app.py)
	$(PY) -m streamlit run app.py

test:             ## run the automated tests
	$(PY) -m pytest -q

lint:             ## static checks
	.venv/bin/ruff check src tests scripts app.py

all: data eda preprocess features baselines gap accuracy graph hybrid phase5 compare maps slides paper report test lint
