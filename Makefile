.PHONY: install run test eval ingest docker
install:
	pip install -r requirements-dev.txt
run:
	uvicorn app.main:app --reload
test:
	pytest -q
eval:
	python -m eval.run_eval
docker:
	docker compose up --build
