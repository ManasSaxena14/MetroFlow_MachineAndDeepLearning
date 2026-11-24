PY := .venv/bin/python

.PHONY: help setup data eda preprocess features baselines gap accuracy graph hybrid ablations compare maps operations dashboard test lint all

help:             ## list the commands
	@grep -E '^[a-z-]+: .*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:            ## create .venv (Python 3.11 via uv) and install the pipeline requirements
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -r requirements-pipeline.txt
	@echo "macOS: XGBoost also needs 'brew install libomp'"

data:             ## download METR-LA and the sensor graph into data/raw
	./scripts/download_data.sh

eda:              ## run the exploratory notebook -> results/findings/exploration.json
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_exploratory_analysis.ipynb

preprocess:       ## leak-free windows -> data/processed/<protocol>/
	$(PY) -m metroflow.data.preprocess

features:         ## per-sensor feature tables
	$(PY) -m metroflow.features.build

baselines:        ## six per-sensor baselines -> results/baselines/
	$(PY) -m metroflow.train.baselines persistence hist_avg ridge xgb rf lstm --peak
	$(PY) -m metroflow.viz.baseline_plots

gap:              ## gap experiments -> results/gap/
	$(PY) -m metroflow.train.gap neighbours onset residuals table
	$(PY) -m metroflow.viz.gap_plots

accuracy:         ## plain-language accuracy scores -> results/baselines/accuracy.csv
	$(PY) -m metroflow.eval.accuracy

graph:            ## STGAT and DCRNN, 3 seeds, ablations (about 10 h on an M3 Pro, resumable)
	./scripts/train_graph_models.sh all

hybrid:           ## DCRNN + XGBoost residual corrector and its variants -> results/hybrid/
	$(PY) -m metroflow.train.hybrid

ablations:        ## ablation table, learning curves, attention inspection -> results/comparison/
	$(PY) -m metroflow.eval.ablations

compare:          ## final comparison and day-level significance tests -> results/comparison/
	$(PY) -m metroflow.eval.compare

maps:             ## congestion maps and time-space diagram -> outputs/figures/
	$(PY) -m metroflow.viz.maps

operations:       ## alerts, forecast ranges, failure tests and the dashboard replay
	$(PY) -m metroflow.train.hybrid_cv
	$(PY) -m metroflow.eval.uncertainty
	$(PY) -m metroflow.eval.alerts
	$(PY) -m metroflow.eval.replay
	$(PY) -m metroflow.eval.robustness --stage graph
	$(PY) -m metroflow.eval.robustness --stage trees

dashboard:        ## open the dashboard locally
	$(PY) -m streamlit run app.py

test:             ## run the automated tests
	$(PY) -m pytest -q

lint:             ## static checks
	.venv/bin/ruff check .

all: data eda preprocess features baselines gap accuracy graph hybrid ablations compare maps operations test lint
