# ===== Makefile for Log-Model Exfil Experiments =====
# Usage examples:
#   make env                     # create venv + install deps
#   make prep DATASET=bgl        # parse, template, window, split
#   make train EXP=wb_exfil_deeplog_bgl SEED=42
#   make attack EXP=wb_exfil_deeplog_bgl RUN=001
#   make defend EXP=wb_exfil_deeplog_bgl DEF=dp_sgd RUN=001
#   make eval MODEL=models/attacked/<model_dir>
#   make clean                   # remove caches and temp files

# --- Configurable vars ---
PYTHON        ?= python
PIP           ?= pip
VENV          ?= .venv
SRC           ?= src
CONFIG_DIR    ?= configs
EXP_DIR       ?= experiments
DATA_DIR      ?= data
MODELS_DIR    ?= models
SEED          ?= 42
RUN           ?= $$(date +"%Y%m%d-%H%M%S")

# respect user-provided CUDA device, else default to 0
CUDA_VISIBLE_DEVICES ?= 3
export CUDA_VISIBLE_DEVICES

# Detect OS for 'find' delete flag
OS := $(shell uname -s)

# --- Phony targets ---
.PHONY: help env freeze prep train attack defend eval sweep report clean deepclean fix-perms

help:
	@echo "Targets:"
	@echo "  prep DATASET=<id>   - data pipeline (raw -> interim -> processed)"
	@echo "  train EXP=<cfg>     - train clean model for experiment cfg"
	@echo "  attack EXP=<cfg>    - train with attack pipeline (white/black-box)"
	@echo "  defend EXP=<cfg>    - apply/train with defense; DEF=<defense_name> optional"
	@echo "  eval MODEL=<path>   - evaluate a saved model (trained/attacked/defended)"
	@echo "  sweep EXP=<cfg>     - run param sweep (requires configs with Hydra)"
	@echo "  report EXP_ID=<id>  - aggregate metrics → reports/"
	@echo "  clean|deepclean     - remove caches, logs, build artefacts"


# --- Data pipeline ---
# DATASET must match a file in configs/dataset/<DATASET>.yaml
prep:
	@[ -n "$(DATASET)" ] || (echo "ERROR: set DATASET=<name>"; exit 1)
	$(PYTHON) -m $(SRC).cli prep --dataset $(DATASET)

# --- Train / Attack / Defend / Eval ---
# EXP must match a file in configs/experiment/<EXP>.yaml
train:
	@[ -n "$(EXP)" ] || (echo "ERROR: set EXP=<experiment_cfg_name>"; exit 1)
	$(PYTHON) -m $(SRC).cli train \
		--config $(CONFIG_DIR)/experiment/$(EXP).yaml \
		--seed $(SEED)

attack:
	@[ -n "$(EXP)" ] || (echo "ERROR: set EXP=<experiment_cfg_name>"; exit 1)
	$(PYTHON) -m $(SRC).cli attack \
		--config $(CONFIG_DIR)/experiment/$(EXP).yaml \
		--seed $(SEED) \
		--run_id $(RUN)

defend:
	@[ -n "$(EXP)" ] || (echo "ERROR: set EXP=<experiment_cfg_name>"; exit 1)
	$(PYTHON) -m $(SRC).cli defend \
		--config $(CONFIG_DIR)/experiment/$(EXP).yaml \
		--defense $(DEF) \
		--seed $(SEED) \
		--run_id $(RUN)

eval:
	@[ -n "$(MODEL)" ] || (echo "ERROR: set MODEL=<path_to_model_dir>"; exit 1)
	$(PYTHON) -m $(SRC).cli eval --model_path $(MODEL)

# --- Optional: sweeps & reports ---
sweep:
	@[ -n "$(EXP)" ] || (echo "ERROR: set EXP=<experiment_cfg_name>"; exit 1)
	$(PYTHON) -m $(SRC).cli sweep \
		--config $(CONFIG_DIR)/experiment/$(EXP).yaml

report:
	@[ -n "$(EXP_ID)" ] || (echo "ERROR: set EXP_ID=<folder_name_in_experiments>"; exit 1)
	$(PYTHON) -m $(SRC).cli report --exp_id $(EXP_ID)

# --- Housekeeping ---
clean:
	@if [ "$(OS)" = "Darwin" ]; then \
	  find . -name "__pycache__" -type d -prune -exec rm -rf {} +; \
	  find . -name "*.pyc" -delete; \
	else \
	  find . -name "__pycache__" -type d -print0 | xargs -0 rm -rf; \
	  find . -name "*.pyc" -print0 | xargs -0 rm -f; \
	fi
	rm -rf *.egg-info .pytest_cache .mypy_cache build dist

deepclean: clean
	rm -rf $(EXP_DIR)/*/logs $(EXP_DIR)/*/artefacts
	@echo "Deep cleaned experiment logs/artefacts (models remain)."

# Fix execution permissions on scripts if needed
fix-perms:
	chmod +x scripts/*.sh 2>/dev/null || true

# --- W&B defaults (override per run) ---
WANDB_PROJECT ?= log-exfil
WANDB_ENTITY  ?= your_wandb_name      # optional
WANDB_MODE    ?= online                # or offline / disabled
WANDB_TAGS    ?=
export WANDB_PROJECT WANDB_ENTITY WANDB_MODE WANDB_TAGS

# login once (uses env var WANDB_API_KEY if set)
wandb-login:
	@if [ -z "$$WANDB_API_KEY" ]; then echo "Set WANDB_API_KEY env var"; exit 1; fi
	$(PYTHON) -c "import wandb,os; wandb.login(key=os.environ['WANDB_API_KEY'])" && echo "W&B logged in."

