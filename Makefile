# Convenience targets. Long targets (train, ablate) take roughly 1.5 h and 1.5 h on a 16-thread CPU.
PY ?= python

.PHONY: install lint test download check-data train ablate evaluate explain demo serve all
install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps
lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
test:
	$(PY) -m pytest -q
download:
	$(PY) -m pxray download
check-data:
	$(PY) -m pxray check-data
train:
	$(PY) -m pxray train --run logreg small_cnn mobilenet_v3_small
ablate:
	$(PY) -m pxray train --run small_cnn_unweighted small_cnn_focal small_cnn_noaug mobilenet_v3_small_scratch --seeds 0 1 2
evaluate:
	$(PY) -m pxray evaluate
explain:
	$(PY) -m pxray explain
demo:
	$(PY) -m pxray export-demo
serve:
	$(PY) -m http.server 8000 --directory docs
all: download check-data train ablate evaluate explain demo
