<!-- Auto-generated: tailored Copilot instructions for this workspace -->
# Copilot / AI agent instructions for this repository

Purpose: Help AI coding agents quickly become productive in this workspace.

1) Big picture
- Repo layout: a small learning notebook project. Primary content lives under `lessonone/` and the working notebook is `lessonone/one.ipynb`.
- Intent: educational/example code (not a production service). Changes usually touch the notebook cells or add small helper scripts next to the notebook.

2) Key files & places to look
- `lessonone/one.ipynb` — canonical source of code, examples and narrative. Inspect the notebook JSON for cell structure but prefer editing via notebook tools when possible.
- Workspace root — may include non-ASCII path segments (e.g., the folder `Eğitim`). Be careful with path-encoding on Windows.

3) How to run & developer workflows (what to try locally)
- Open the workspace in VS Code and open `lessonone/one.ipynb` in the Notebook editor.
- To run interactive cells use the VS Code Jupyter experience or `jupyter nbconvert --to notebook --execute lessonone/one.ipynb --inplace` for headless execution.
- If you need a Python environment, create a virtual env and install packages that appear in notebook imports. Use `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; pip install -r requirements.txt` if a `requirements.txt` is added.

4) Editing guidelines for AI agents
- Prefer non-destructive edits: when possible, add new cells (or append new files such as `lessonone/one.py`) rather than rewriting existing notebook structure.
- If patching the `.ipynb` JSON directly, keep JSON formatting stable: change only the minimal fields required (cell source/output/metadata) and avoid reordering existing cells unless explicitly requested.
- For code changes that should be runnable as scripts, create a small `.py` alongside the notebook (e.g., `lessonone/run_demo.py`) and demonstrate usage with a short README entry.

5) Project-specific conventions & caveats
- Paths contain non-ASCII characters (workspace folder `Eğitim`). On Windows, prefer using Python's `pathlib` for reliable path handling and avoid hardcoding encodings.
- This repo currently has no tests, build system, or requirements file — treat it as a lightweight demo workspace.
- Keep edits minimal and clearly focused: this is an educational notebook, so prefer explanatory text and small runnable examples.

6) Integration points & external dependencies
- No explicit external services are referenced in the repository. Dependencies will be discoverable only from imports inside notebook cells — inspect cells to infer packages to install.

7) How AI agents should produce patches
- Use `apply_patch` to create or update files. When updating the notebook, prefer adding a `.py` helper file or a new markdown cell rather than wholesale `.ipynb` rewrites.
- Include concise commit messages describing intent (e.g., "Add helper script to demonstrate X" or "Update notebook with clarified explanation for Y").

8) Examples (concrete suggestions)
- To add a small runnable example: create `lessonone/run_demo.py` with a `main()` that runs the core code from the notebook and a short README section showing how to run it.
- To execute headlessly (CI-friendly): `jupyter nbconvert --to notebook --execute lessonone/one.ipynb --inplace`

9) When to ask for clarification
- If edits require adding project-level config (tests, CI, requirements), ask the maintainer before making that decision.

Please review these notes. If anything important is missing — for example, a target Python version, preferred commit message style, or additional notebooks — tell me and I will update this file.
