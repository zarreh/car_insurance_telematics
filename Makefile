lint: ## Lint and reformat the code
	@poetry run autoflake car_insurance_telematics --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py
	@poetry run black car_insurance_telematics --line-length 120 -q
	@poetry run isort car_insurance_telematics