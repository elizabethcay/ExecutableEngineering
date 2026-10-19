---
trigger: always_on
description: "Description of the formatting rule"
---

# Agent Rules

- **Stack**: Jupyter Book `==2.1.6` (MyST ecosystem). Config: `myst.yml`, `toc.yml`. No `_config.yml`/`_toc.yml`. Do NOT downgrade jupyter-book.
- **Monorepo**: Contains book (`Chapters/`, `index.md`) & package (`executable_engineering`).
- **Folder Structure**: When creating a new section or chapter directory, the introductory file for that section must be placed *inside* its corresponding subfolder (e.g., `direct_methods/direct_methods.ipynb`), not outside in the parent directory. This keeps component logic modular and prevents filename collisions.
- **File Modification**: Use built-in `write_to_file` or `replace_file_content` tools to modify files instead of using terminal commands like `cat`.
- **Package**: PyPI & local name is `executable_engineering`. Local dev: `-e ./executable_engineering`. Colab: `pip install executable_engineering`. Build: `python -m build ./executable_engineering`.
- **Interactivity**: Core philosophy: interactive numerical methods. PREFER `plotly` for all plotting. Use `ipywidgets`, `numpy`, `scipy`. (Use `matplotlib` only when static output is strictly required).
- **Cross-Environment Execution**: 
  - All pages must include a Colab markdown badge immediately below the main `# Title` in the first markdown cell. To prevent MyST from auto-centering the badge as a block figure, always prepend it with `&nbsp;`, i.e., `&nbsp;[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ExecutableEngineering/blob/main/...)`
  - Do not assume packages are pre-installed in the cloud. Avoid heavy C-compiled dependencies unsupported by Pyodide/JupyterLite.
  - Only add the setup block when necessary (i.e., when Python code is actually used on the page). When needed, all required `import` statements (e.g., `numpy`, `plotly`) MUST be grouped together in a single code block located directly after the main `# Title` and Colab badge. Only include packages that are actually used in the page. This block must include the `executable_engineering` setup (if used) and must NOT be hidden. Example:
    ```python
    import numpy as np
    import plotly.graph_objects as go
    
    try:
        import executable_engineering as exe
    except ImportError:
        %pip install -q executable_engineering
        import executable_engineering as exe
    ```
- **Git**: NEVER commit `_build/`, `.venv/`, `.ipynb_checkpoints/`, or `__pycache__/`.
- **CI/CD**: GitHub Actions deploys to GitHub Pages from `main`. Do not commit built HTML files.
- **License**: Content = CC-BY-NC-4.0. Code = MIT.

### Course Context & Math
- **Nature of Course**: Numerical Methods for undergraduate engineers. Focus on algorithmic thinking, computational efficiency, and executable code mirroring math.
- **Math Formatting**: Use robust, standard MathJax (avoid custom macros).
  - *Vectors*: `\mathbf{x}`. *Matrices*: `\mathbf{A}`. *Scalars*: `x`.
  - *Derivatives/Operators*: Upright Roman (`\mathrm{d}x`).
  - *Equations*: DO NOT use MyST equation numbering `(label)`. Do not number equations unless specifically requested. Ensure they have horizontal scroll if at all possible.
  - *Equation Arrays*: Use `\begin{aligned} ... \end{aligned}` inside `$$ ... $$` for robust multi-line alignment. Do not use the obsolete `eqnarray`.
  - *File Naming*: All files and directories within the book content (e.g., inside Chapters/) must be named using `snake_case` (all lowercase, no spaces). This ensures maximum compatibility with command-line tools and URLs.
  - *Page Titles*: Every Markdown or Jupyter Notebook file must contain exactly one top-level header (e.g., `# Title`) at the very top of the document. Jupyter Book uses this H1 header for the sidebar index and page title.

### Pedagogical Style Guide (Colab Presentation Mode)
- **Chunking for Colab Slides**: In Google Colab, every heavy Markdown header (`#` or `##`) triggers a new presentation slide. Furthermore, Colab only supports displaying *a single markdown cell and a single code cell* together on one slide. 
- **Notebook Cells**: To ensure slides are complete and effective, group all text for a single slide into exactly ONE markdown cell, placing the heavy header at the top of that cell. If the slide requires code, place exactly ONE code cell immediately following that markdown cell. Do not fragment text across multiple markdown cells within the same section, as Colab will not display them together. Keep markdown concise, focusing on intuition rather than walls of text. Minimal removal of text or comments is allowed to improve flow, correctness, or slide chunking, but key pedagogical points and exploratory code (e.g., printing matrices, checking condition numbers) MUST be preserved.
- **Code Simplicity & Readability**: Keep Python code as simple and concise as possible so it fits cleanly on a single presentation slide without scrolling.
- **No Docstrings**: Forgo standard documentation and docstrings in the notebooks unless absolutely necessary for the lesson.
- **Abstraction**: Abstract heavy boilerplate (like plotting setup) into the `executable_engineering` Python package to keep the notebook focused on the core math/concept.
- **Focus**: Emphasize comprehension over rote memorization. Target audience: 2nd-year undergraduate engineering physics students. Rely heavily on geometric interpretation and visual proofs.