.PHONY: install test lint eval eval-ablation run dev deploy

install:
	pip install -e ".[dev,workspace]"

lint:
	ruff check .

test:
	pytest -q

eval:
	EMBEDDER=hashing python -m evals.run_eval

eval-ablation:
	python -m evals.run_eval --ablation

run:
	uvicorn app.server:app --reload --port 8080

dev:            ## ADK developer UI (traces, events, state) at :8000
	adk web adk_agents

deploy:         ## build with Cloud Build, deploy to Cloud Run
	gcloud builds submit --config deploy/cloudbuild.yaml --substitutions=_REGION=$${REGION:-asia-south1}
