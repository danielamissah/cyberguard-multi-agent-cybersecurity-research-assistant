PYTHON = /opt/anaconda3/bin/python3
PYTHONPATH = PYTHONPATH=.

.PHONY: setup kb api dashboard test lint

setup:
	pip install -r requirements.txt

kb:
	$(PYTHONPATH) $(PYTHON) src/tools/knowledge_base.py

api:
	$(PYTHONPATH) uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload

dashboard:
	$(PYTHONPATH) streamlit run dashboard/app.py

test:
	$(PYTHONPATH) pytest tests/ -v

lint:
	ruff check src/ --ignore E501

query:
	$(PYTHONPATH) $(PYTHON) src/graph/agent_graph.py
