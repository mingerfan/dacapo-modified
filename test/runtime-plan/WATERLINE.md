# Waterline export and checks

Use `runtime-plan-plaintext-schedule=waterline` after placement and communication
materialization. `plan-plaintext-waterline` rewrites direct Host Encode uploads
into `dist.prepare_plain`. The pass runs before runtime memory planning; no CSE
or motion pass should follow the lifecycle rewrite. It does not change HEFT.

Configure total per-device capacity, a separate persistent/safety reserve,
waterline fraction, shared Host/pinned/raw budgets, complete backend (`cpu`,
`gpu`, `auto`), and lookahead groups. The scalar compiler fraction initializes
all devices; V4 device tables can configure each card separately at plan loading.
The workspace option is a backend upper bound for the compiler capacity precheck.
The runtime checks the concrete backend bound before submitting online work.

The former `stream` mode, static batch cloning and automatic Fence insertion
are removed. Regenerate those experimental plans. A manually authored Fence
continues to mean completion of already submitted work at a real boundary.

```sh
python3 test/runtime-plan/check-waterline.py \
  --hecate-opt build/nix/bin/hecate-opt --source-dir .
```

The test checks V4 export, a shared Prepare without reencoding, preserved
foreground devices/order, no automatic Fence, and impossible single-group
capacity/waterline/workspace rejection.
