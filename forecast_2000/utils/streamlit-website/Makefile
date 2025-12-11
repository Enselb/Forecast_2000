.DEFAULT_GOAL := help
PACKAGE = forecast_2000
reinstall_package:
	@Pip uninstall -y $(PACKAGE) || :
	@Pip install -e .
run_app:
	streamlit run forecast_2000/app.py
run_train:
	python -m forecast_2000.training  # ou le chemin réel de ton script de train
test:
	pytest
clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr *.dist-info
	@rm -fr *.egg-info
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
help:
	@ECho "Cibles disponibles :"
	@ECho "  make reinstall_package  - réinstalle le package en mode editable"
	@ECho "  make run_app            - lance l'app Streamlit"
	@ECho "  make run_train          - lance le script de training (à adapter)"
	@ECho "  make test               - lance les tests pytest"
	@ECho "  make clean              - nettoie le repo"
