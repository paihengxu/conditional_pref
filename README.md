# Python Project Template

A clean, modular, and lightweight Python project template based on the development patterns used in `text_diff`.

## Directory Structure

```text
├── .env.example            # Environment variables placeholder (e.g. API keys)
├── .gitignore              # Standard git ignores (including caches and ignored repos)
├── README.md               # Project documentation
├── requirements.txt        # Minimalist package dependencies
├── setup.py                # Setup script for editable package installation
├── configs/
│   └── config.yaml         # Project configuration file
├── data/                   # Data files storage
├── docs/                   # Project documentation / papers
├── notebooks/              # Jupyter notebooks for exploration
├── references/             # Explanatory materials / research papers
├── repos/                  # External cloned repositories folder (ignored from main git history)
├── scripts/
│   └── run_pipeline.py     # Executable sample entrypoint / workflow pipeline
├── src/
│   └── my_package/         # Your actual code package
│       ├── __init__.py     # Package initialization
│       ├── config.py       # Configuration loader and directory resolvers
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py   # Dataset loading, splitting, and bundling boilerplate
│       └── utils/
│           ├── __init__.py
│           ├── cache.py    # Thread-safe local JSONL raw completion cache
│           ├── llm_api.py  # OpenAI client, completions, retries, and chunked embedding cache
│           ├── llm_local.py# Offline vLLM loader and GPU memory manager (sleep/wake)
│           └── text.py     # Clean text truncation helpers (chars, words, tokens)
├── tests/
│   ├── __init__.py
│   └── test_utils.py       # Modular suite of unit tests
├── auto_task/              # Periodic automation agent orchestration (see auto_task/README.md)
│   └── ...
└── slurm_bridge/           # Shared-filesystem control-action bridge (see slurm_bridge/README.md)
    └── ...
```

---

## Getting Started

### 1. Environment Setup
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` to supply your credentials:
   ```bash
   OPENAI_API_KEY=your-actual-api-key-here
   ```

### 2. Dependency Installation
Install the core minimal dependencies:
```bash
pip install -r requirements.txt
```

If you plan to run **local offline models** (using `llm_local.py`) or perform **complex data analysis**, uncomment and install the heavy libraries (e.g., vllm)

### 3. Package Installation
Install the project code in editable development mode:
```bash
pip install -e .
```
This maps imports of `my_package` straight to the `./src/my_package` folder, allowing changes to take effect immediately across any folder/notebook.

---

## Running the Pipeline

You can run the sample entrypoint to verify configurations and LLM calls function correctly out of the box:
```bash
python scripts/run_pipeline.py
```

---

## Running Tests

Run the standard testing suite via Python's built-in unittest framework:
```bash
python -m unittest discover -s tests
```

---

## Periodic Automation & Slurm Bridge

This project template includes production-ready utilities to run continuous coding agent iterations and execute Slurm workloads across a login/compute node boundary.

Detailed setup and workflows for these utilities are stored under their respective directories:
- **Periodic Automation (`auto_task/`)**: See [auto_task/README.md](auto_task/README.md) for continuous agent orchestration using cron and file-based task management.
- **Slurm Bridge (`slurm_bridge/`)**: See [slurm_bridge/README.md](slurm_bridge/README.md) for executing scheduler control commands across compute/login nodes.
