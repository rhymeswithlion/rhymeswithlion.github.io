.venv:
	@if [ ! -f ./.venv/bin/python3.11 ]; then \
		conda create -p ./.venv python=3.11 -y; \
		conda run -p ./.venv python -m venv ./.venv; \
	fi
	./.venv/bin/pip install -q -r requirements.txt
	@echo "Virtual environment ready and requirements installed."
	@echo "To activate this environment, use:"
	@echo "source ./.venv/bin/activate"