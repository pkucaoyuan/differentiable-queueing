# Blocked: GPU Experiments

Last verified: 2026-05-15

---

## Summary

**The paper's GPU acceleration claim cannot be reproduced or verified.** Multiple blockers:

1. PyTorch installed is `2.2.0+cpu` build — no CUDA support compiled in
2. The `gpu.q` scheduler queue contains only `researchgpu03`
3. `researchgpu03` is permanently saturated (24/24 slots used, 7+ jobs queued)
4. `researchgpu04` and `researchgpu05` have hardware (8×A40 each) but are NOT in any SGE queue
5. We have no documented access mechanism for gpu04/05

---

## What we tried

### Attempt 1: torch with CUDA build
```bash
$ python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
2.2.0+cpu
False
```

To install CUDA-enabled torch we'd need:
```bash
pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

But this is pointless without GPU compute access.

### Attempt 2: Check GPU queue availability
```bash
$ qstat -q gpu.q -u '*'
job-ID  prior   name       user         state submit/start at     queue                          slots
8558749 0.55036 run_bootst as5443       r     04/20/2026 10:40:01 gpu.q@researchgpu03.grid.gsb      16
8543693 0.55244 TEST_NG_ME av3334       qw    03/20/2026 15:42:25                                   24
... (7 more jobs in qw state)
```

gpu03 is the only node and is always full.

### Attempt 3: Check gpu04/gpu05
```bash
$ qhost -h researchgpu01,researchgpu02,researchgpu03,researchgpu04,researchgpu05 -q
HOSTNAME                ARCH         NCPU  LOAD  MEMTOT  MEMUSE
researchgpu01           lx24-amd64     24     -  187.4G       -    (offline)
researchgpu02           lx24-amd64     24  0.99  187.4G   45.5G    (online but not in any queue)
researchgpu03           lx24-amd64     24  1.24  376.4G  107.3G
   gpu.q                BIP   0/16/24       
researchgpu04           lx24-amd64    128 27.36 1007.8G   31.6G    (has load — someone using directly?)
researchgpu05           lx24-amd64    128     - 1007.8G       -    (offline)
```

`@gpu` host group only contains `researchgpu03.grid.gsb`. gpu04/05 hardware exists but is not exposed to scheduler.

### Attempt 4: Check env.py GPU path
The `queuetorch/env.py` was modified to support GPU-native sampling via `torch.distributions`. Code path looks correct but never executed:

```python
$ python -c "
import torch
from queuetorch.env import load_env
import yaml
with open('configs/env/mm1.yaml') as f: cfg = yaml.safe_load(f)
dq = load_env(cfg, temp=0.1, batch=10, seed=42, device='cuda')
"
AssertionError: Torch not compiled with CUDA enabled
```

---

## Impact on reproduction

### What we CAN reproduce
- All paper experiments (5.1, 5.2, 5.3, 6, 7, 8) — paper ran these on CPU too
- Our reproductions and revision experiments all on CPU produce correct results

### What we CANNOT reproduce
- **Paper's GPU acceleration claim** (Abstract, Section 4 references)
- E3 revision experiment (GPU wall-clock benchmarks for AE/Referee 2)

### Important observation
The paper itself never validates GPU acceleration with experiments. The abstract claim is unsupported. This is exactly what AE Major Comment 4 and Referee 2 are calling out for the revision.

---

## What's needed to unblock

Either:
1. **Get gpu04/gpu05 added to the SGE queue** — email ResearchSupport@gsb.columbia.edu
2. **Get permission to SSH directly to gpu04/gpu05** — non-standard for this cluster
3. **Use another GPU resource** — Columbia CUIT cluster, AWS, etc.

For E3 specifically, gpu04 has 8×A40 (48GB each) which would be ideal for the benchmark — even one A40 is sufficient.

---

## Action items

- [ ] Send email to ResearchSupport@gsb.columbia.edu asking about gpu04/05 access
- [ ] Discuss with advisor: is GPU validation a hard requirement, or can we proceed with CPU + textual argument about GPU compatibility (no custom CUDA kernels needed)?
