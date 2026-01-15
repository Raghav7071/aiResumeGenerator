.PHONY: setup backend frontend run docker clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@[ -f .env ] || cp .env.example .env && echo "Created .env — add your GROQ_API_KEY"

run:
	uvicorn backend.main:app --reload --port 8000

docker:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf venv
