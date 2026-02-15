# Agent guide: rhymeswithlion.github.io

This repo is a **Quarto-based blog/site** for GitHub Pages (i.stolethis.website). Use this file when helping with content, build, or publishing.

## Stack and tooling

- **Quarto** (website): config in `_quarto.yml`; content in `.qmd` and `.ipynb`.
- **Python/uv**: venv in `.venv`; dependencies in `pyproject.toml` and `uv.lock`. Quarto is provided by the **quarto-cli** pip package.
- **Makefile**: `make .venv` → create venv and sync deps; `make preview` → local preview; `make render` → build to `_site/`.

## Key paths

| Path | Role |
|------|------|
| `_quarto.yml` | Site config: project type, navbar, theme, output. |
| `index.qmd` | Homepage; lists contents of `posts/` (date desc). |
| `about.qmd` | About page. |
| `posts/` | Blog posts. Only non-draft posts appear in the listing. |
| `posts/drafts/` | Draft posts: `_metadata.yml` has `draft: true` and `draft-mode: unlinked` so they don’t appear in the main listing. |
| `_site/` | Rendered output (do not edit by hand). |
| `pyproject.toml` / `uv.lock` | Python deps; use `uv add …` to add packages. |

## Workflow for agents

1. **Setup**: Ensure environment exists — run `make .venv` (or `uv sync`).
2. **Preview**: Run `make preview` to serve the site and watch for changes.
3. **Build**: Run `make render` to regenerate `_site/`.
4. **Publishing**:
   - **New post**: Add `.qmd` or `.ipynb` under `posts/` (or under `posts/drafts/` and set `draft: true` in a `_metadata.yml` there).
   - **Publish a draft**: Move the post out of `posts/drafts/` into `posts/`, or remove/override `draft: true` and adjust `draft-mode` so it appears in the listing.
   - **Deploy**: Push to the repo; GitHub Pages serves from the configured branch/source (output is in `_site/` unless `_quarto.yml` overrides it).

## Conventions

- Use the project’s **uv** and **Makefile** for env and Quarto; don’t assume a system-wide `quarto` or a different package manager.
- Don’t edit files under `_site/` or `.quarto/`; they are generated.
- Draft content lives under `posts/drafts/` with `draft: true` and is unlinked until moved or metadata is updated.
