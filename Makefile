install:
	pip install -e .

prepare-input:
	python scripts/preprocess/prepare_plot_input.py --config configs/plot_area.yaml

run-embeddings:
	python scripts/run_plot_area_embeddings.py --config configs/plot_area.yaml

run-clean: install prepare-input run-embeddings