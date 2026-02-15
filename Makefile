.venv:
	uv sync
	@echo "Virtual environment ready and dependencies synced (includes quarto-cli)."
	@echo "To activate this environment, use:"
	@echo "  source ./.venv/bin/activate"

preview: .venv
	source ./.venv/bin/activate && quarto preview

render: .venv
	source ./.venv/bin/activate && quarto render

.PHONY: .venv preview render
