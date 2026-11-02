.PHONY: install run demo test bench docker
install: ; pip install -r requirements.txt
run: ; uvicorn api.main:app --reload --port 8000
demo: ; python -m scripts.demo
test: ; python -m unittest discover -s tests -v
bench: ; python -m scripts.benchmark
docker: ; docker compose up --build
