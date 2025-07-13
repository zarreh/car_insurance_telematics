lint: ## Lint and reformat the code
	@poetry run autoflake car_insurance_telematics --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py
	@poetry run black car_insurance_telematics --line-length 120 -q
	@poetry run isort car_insurance_telematics

train:
	@poetry run python -m car_insurance_telematics.modeling.train_models

infer:
	@poetry run python -m car_insurance_telematics.modeling.run_inference --use-sample-data

infer-bathch:
	@poetry run python -m car_insurance_telematics.modeling.run_inference --input-file ./data/processed/processed_trips_1200_drivers.csv