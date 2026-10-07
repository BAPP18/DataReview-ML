PYTHON ?= python

.PHONY: install test lint format check run validate docker-up docker-down

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	ruff check .

format:
	ruff format .

check:
	ruff check .
	$(PYTHON) -m unittest discover -s tests -v

run:
	streamlit run app/dashboard.py

validate:
	$(PYTHON) validate_env.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down
