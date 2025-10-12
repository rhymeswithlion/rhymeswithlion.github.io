.venv:
	@if [ ! -f ./.venv/bin/python ]; then \
		uv venv ./.venv --python 3.11; \
	fi
	uv pip install -r requirements.txt
	@echo "Virtual environment ready and requirements installed."
	@echo "To activate this environment, use:"
	@echo "source ./.venv/bin/activate"