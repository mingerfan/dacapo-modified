# Plaintext streaming

All RuntimePlan export pipelines accept `--runtime-plan-plaintext-schedule=eager|stream`
(default eager). Stream requires placement on one rank and positive byte budgets:

- `--runtime-plan-gpu-budget-bytes`: RNS object capacity per card after reserves.
- `--runtime-plan-host-budget-bytes`: encoded Host objects, including ciphertexts.
- `--runtime-plan-pinned-budget-bytes`: Host/Device transfer staging within a batch.
- `--runtime-plan-raw-budget-bytes`: one blob buffer and decoded slots.
- `--runtime-plan-workspace-bytes-per-op`: cumulative GPU temporary reservation.
- `--runtime-plan-prefetch-bytes`: optional byte-bounded batch preparation prefix.

PlanPlaintextStreaming runs after MaterializeCommunication and before
PlanRuntimeMemory. It scans uses once, preserves non-weight order, duplicates
Encode/direct uploads per batch, then inserts effectful dist.fence boundaries.
Memory planning is always enabled in stream mode. Do not run Encode CSE afterward.
The pass reports an indivisible operation and its device/required bytes if it
cannot fit; there is no eager fallback. Plaintext final outputs and multiple
ranks are currently rejected.

The streaming report counts batch-start live objects plus all new allocations
and never returns Release credit until Fence. Reuse is conservatively charged
as a new allocation in this report. Pinned copies and raw reads have separate
budgets. Unknown workspace, keys, parameters, encoder workspace, pool retention
and fragmentation are excluded; total_device_memory_bound is false. Reserve
capacity before choosing budgets and measure the actual run.

The optional prefetch prefix uses bytes, not instruction distance. Latency-aware
scheduling and event-based admission are future optimizations. No CUDA allocation
readiness wait is removed. `check-streaming.py` independently simulates retained
allocations, checks all budget failures and executes a 20,000-operation case.
