# Source study template

One note per kernel group or architecture cell. Keep it short; the code and the
registry row are evidence, this file records what was read and what differs.

## Identity

- Kernel(s):
- Family / step:
- Arch body:
- Owner / date:

## Upstream sources

| Role | Repository | Commit | Files |
|---|---|---|---|
| Algorithm / reference | | | |
| Layout / codec | | | |
| Scheduling / pipeline | | | |

## Contract extracted

- Inputs / outputs / strides:
- Numerical order and rounding points:
- LSE / sink / mask convention:
- KV codec and page padding:
- Graph and workspace rules:

## Differences from upstream

| Topic | Upstream | kernel-ops | Why |
|---|---|---|---|
| | | | |

## Twin T2 coverage

- Scheduler / raster:
- Address layout / swizzle:
- mbarrier stage/phase parity:
- Warp roles:
- Epilogue / scale / mask:
- Not covered (TMA/TMEM/cluster/perf):

## Test and golden vector

- Unit test:
- Golden source:
- Tolerance:
- Handoff command:

## Upgrade ideas

- [ ] Fusion:
- [ ] Traffic reduction:
- [ ] Determinism / batch invariance:
- [ ] Graph replay:

## Provenance / license

- Files ported (not merely read):
- `THIRD_PARTY_NOTICES.md` updated: yes/no
