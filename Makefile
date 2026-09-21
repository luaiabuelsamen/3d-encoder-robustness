# Reproduce the study end to end.
#
#   make check     geometry and renderer self-tests (run these first, always)
#   make data      collect train/val/test demonstrations
#   make train     train every arm at every seed
#   make grid      evaluate every checkpoint across the stress grid
#   make figures   figures and results table
#   make all       the lot
#
# A system ROS install registers pytest plugins globally that fail on a missing
# `lark`, hence PYTEST_DISABLE_PLUGIN_AUTOLOAD.

# PYTHONPATH is deliberately left alone. The shell profile on this host exports
# one that selects a working CUDA 12.6 torch build; replacing or unsetting it
# selects a CUDA 13.0 wheel instead and CUDA silently disappears. The scripts
# put the repo on sys.path themselves. See rvt_lerobot/device.py.
PY       ?= MUJOCO_GL=egl python
DATA     ?= data
RUNS     ?= runs
RESULTS  ?= results
SEEDS    ?= 0 1 2
ARMS     ?= proprio rgb rgbd rgbd_unproj xyz_real rvt rgb_aug rvt_aug
TRAIN_EP ?= 500
VAL_EP   ?= 60
TEST_EP  ?= 120
STEPS    ?= 8000
IMAGE    ?= 96

export PYTEST_DISABLE_PLUGIN_AUTOLOAD = 1

.PHONY: all check data train grid figures clean

all: check data train grid figures

check:
	$(PY) scripts/check_conventions.py
	$(PY) scripts/check_virtual_views.py
	$(PY) scripts/smoke_arms.py

data: $(DATA)/train.npz $(DATA)/val.npz $(DATA)/test.npz

$(DATA)/train.npz:
	$(PY) scripts/collect.py --out $@ --num $(TRAIN_EP) --seed 0
$(DATA)/val.npz:
	$(PY) scripts/collect.py --out $@ --num $(VAL_EP) --seed 100000
$(DATA)/test.npz:
	$(PY) scripts/collect.py --out $@ --num $(TEST_EP) --seed 200000

train: data
	@for arm in $(ARMS); do \
	  for s in $(SEEDS); do \
	    if [ -f $(RUNS)/$$arm\_s$$s/model.pt ]; then \
	      echo "skip $$arm seed $$s (done)"; \
	    else \
	      $(PY) scripts/train.py --arm $$arm --seed $$s --steps $(STEPS) \
	        --image $(IMAGE) --episodes $(DATA)/train.npz \
	        --val-episodes $(DATA)/val.npz --out $(RUNS) || exit 1; \
	    fi; \
	  done; \
	done

grid:
	$(PY) scripts/eval_grid.py --episodes $(DATA)/test.npz --runs $(RUNS) \
	  --out $(RESULTS)/grid.json --image $(IMAGE)

figures:
	$(PY) scripts/make_figures.py --grid $(RESULTS)/grid.json --out figures

clean:
	rm -rf $(RUNS) $(RESULTS)
