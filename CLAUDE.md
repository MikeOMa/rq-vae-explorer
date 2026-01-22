# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is set up with Ralph Orchestrator for agentic workflows using Claude as the backend.

Please research python package apis where needed:


## Ralph Orchestrator

Configuration is in `ralph.yml`. The event loop expects:
- A `PROMPT.md` file containing the task description
- Completion signaled by `LOOP_COMPLETE` promise

To run an agentic task:
```bash
ralph run
```

## Commands

```bash
# Install dependencies
uv sync

# Run the app
uv run rq-vae-explorer

# Run tests
uv run pytest tests/ -v

# Run single test
uv run pytest tests/test_file.py::test_name -v
```

## Architecture

- `src/rq_vae_explorer/model/` - Flax-based RQ-VAE (encoder, decoder, quantizer)
- `src/rq_vae_explorer/training/` - Training loop, losses, thread-safe state
- `src/rq_vae_explorer/data/` - MNIST data loading
- `src/rq_vae_explorer/ui/` - Gradio application and plotting
