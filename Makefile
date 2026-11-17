.PHONY: check figures supplement notebook smoke experiment
check:
	python scripts/verify_artifact.py
figures:
	python scripts/regenerate_figures.py
supplement:
	python scripts/build_supplement.py
notebook:
	python scripts/build_notebook.py
smoke:
	python scripts/reproduce.py --profile smoke
experiment:
	python scripts/reproduce.py --profile paper
