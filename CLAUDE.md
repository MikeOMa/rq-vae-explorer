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
