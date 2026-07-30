# Backpressure — design decisions & refactor record

> **Status: IMPLEMENTED.** This started as a proposal; it has now been carried out.
> This document records what changed, why, and how each change maps back to the
> mechanism explained in [`README.md`](./README.md). Read the README first for the
> mental model (three gauges, one rule, two flows, fail-open).
>
> **All changes were behavior-preserving** — the runtime decision path a request
> experiences is unchanged. These were structural, naming, and clarity changes:
> dead code removed, a type-checker shim deleted, the probes grouped into a
> sub-package, and one mislabeled layer number corrected. Verified by the package
> test suite (16 passing) plus an import check of every module.

---

## 1. Why we touched it

The module was already well-architected — reading, deciding, and presenting were
cleanly separated (README §9). But three pieces of clutter sat on top of the good
design and hid it from a first-time reader:

1. A **dead compatibility class** presenting a fake second entry point.
2. A **type-checker shim** that buried real logic under scaffolding.
3. A **"Layer 1" vs "Layer 2" contradiction** across files.

We removed exactly those obstacles and grouped the probes so the folder now reads
as cleanly as the model the README teaches.

---

## 2. Change 1 — deleted the dead facade `backpressure_gate.py` ✅

**Correlates to:** README §2 (the real entry point) and §5 (the front door).

`backpressure_gate.py` defined `BackPressureGuard`, self-described as a
*"compatibility facade for legacy callers."* A full-codebase search proved no
caller ever invoked its `.check()` / `.evaluate()` / `.as_dependency()` methods —
the only references were re-exports and a test asserting the symbol existed. It
presented a confusing *second* front door next to the real one, and its file name
(`gate`) even disagreed with its class name (`Guard`).

**Done:**
- Deleted `backpressure_gate.py`.
- Removed `BackPressureGuard` from `backpressure/__init__.py` and
  `app/resilience/__init__.py`.
- Simplified `tests/.../test_public_api.py` to pin the one real export,
  `backpressure_dependency`.

**Result:** exactly one obvious front door.

---

## 3. Change 2 — removed the type-checker shim entirely ✅

**Correlates to:** README §5 (probes/publisher "read one signal, return a clean
number") and §6 (fail-open — logic must be legible to be trustworthy).

The publisher and the pool probe carried `typing.Protocol` classes (and a separate
`_broker_typing.py` module) that existed *only* to describe untyped third-party
objects (Kombu, SQLAlchemy) back to the static checker. They had **zero runtime
effect** but dominated the files and read like a hack — because they were one.

The original proposal was to *relocate* them. We went further and **deleted them**,
rewriting the code to use the libraries directly:
- `publisher.py` now uses plain nested `with Connection(...) as c: with c.channel()
  as ch:` — obvious, and the resource cleanup is visible.
- `probes/db_pool.py` now calls `pool.size()` / `pool.checkedout()` directly.
- `_broker_typing.py` was removed.

**Result:** each file opens with its actual job; nothing exists just to appease a
tool. (SQLAlchemy is typed, so the pool probe needs no shim at all; Kombu is
untyped, which is fine — the plain code is clearer than a fake interface.)

---

## 4. Change 3 — fixed the "Layer" label to Layer 1 ✅

**Correlates to:** README §2 (the note on naming).

The module headers said *"Layer 1"*; the endpoint comment said *"Layer 2."* We
resolved it from the canonical source of truth rather than guessing:
`SYSTEM_DESIGN.md` and the `documentation/resilience_architecture.md` table both
define **Backpressure = Layer 1** (Layer 0 = rate limiter, Layer 2 = circuit
breaker). So the module was right; the endpoint was wrong.

**Done:** corrected the two "Layer 2" mentions in
`app/api/token_manager_endpoints.py` to "Layer 1." Every file now agrees.

---

## 5. Change 4 — added in-folder documentation ✅

`README.md` (mental model + mechanics) and this document now live beside the code,
so the reasoning travels with it. Every source file also carries teaching-grade
comments explaining its imports, algorithm steps, config, and decisions.

---

## 6. Change 5 — adopted the `probes/` sub-package ✅

**Correlates to:** README §9 (structure that mirrors the model).

The original proposal *deferred* this (three files felt below the clustering
threshold). That call was overridden deliberately: grouping the three gauges makes
the directory tree teach the "read → decide → translate" flow at a glance, which
serves the primary goal — comprehension.

**Done:** created the `probes/` package and moved/renamed the readers:

| Was | Now |
|---|---|
| `token_queue_depth_probe.py` | `probes/queue_depth.py` |
| `db_connection_pool_probe.py` | `probes/db_pool.py` |
| `circuit_state_probe.py` | `probes/circuit_state.py` |

`probes/__init__.py` re-exports the three reader functions, so the evaluator (and
any caller) imports them from one place. All importers, tests, and docs were
updated to the new paths.

---

## 7. Naming review — what changed, what stayed

**Correlates to:** README §5 and §9.

Most names were already good and were **kept**. Only the genuinely misleading or
verbose ones changed:

| Symbol / file | Verdict | Note |
|---|---|---|
| `evaluate_backpressure()` | keep | Says what it does |
| `read_queue_depth()`, `estimate_queue_retry_after_seconds()` | keep | Clear |
| `read_db_pool_utilization_pct()` | keep | Units in the name |
| `read_db_circuit_breaker_snapshot()` | keep | Clear, read-only intent |
| `raise_for_backpressure_decision()` | keep | Mirrors `raise_for_status()` |
| `backpressure_dependency()` | keep | The one real front door |
| `publish_queue_depth_snapshot()` | keep | Clear Flow-B writer |
| `backpressure_gate.py` / `BackPressureGuard` | **deleted** | Dead code |
| `backpressure_http_response.py` | **renamed** → `http_response.py` | Shorter, unambiguous in-package |
| `queue_depth_publisher.py` | **renamed** → `publisher.py` | Shorter, unambiguous in-package |
| the three `*_probe.py` files | **moved** → `probes/*.py` | Grouped; the `probes/` package name now carries the "probe" meaning |

---

## 8. Final folder layout

```
backpressure/
  README.md                  # mental model + mechanics
  PROPOSED_DESIGN.md         # this record
  __init__.py                # exports backpressure_dependency only
  constants.py               # shared Redis key + reason codes
  dependency.py              # THE front door (FastAPI Depends target)
  evaluator.py               # the rule: 3 gauges, first-red-wins
  http_response.py           # decision → 503 translator
  publisher.py               # Flow B writer (background)
  probes/
    __init__.py              # re-exports the three readers
    queue_depth.py           # gauge #1 (+ Retry-After estimate)
    db_pool.py               # gauge #2
    circuit_state.py         # gauge #3
```

---

## 9. Change summary

| # | Change | Behavior impact | Status |
|---|---|---|---|
| 1 | Delete `backpressure_gate.py` + re-exports + dead test asserts | none | ✅ done |
| 2 | Remove the type-checker shim; use libraries directly | none | ✅ done |
| 3 | Fix the "Layer" label to Layer 1 (endpoint corrected) | none (comment) | ✅ done |
| 4 | Add README + this doc + in-code teaching comments | none | ✅ done |
| 5 | Adopt the `probes/` sub-package | none | ✅ done |

**Verification:** the four collectable backpressure test files pass (16 tests);
every refactored module imports cleanly; the public entry point identity-checks
against its parent re-export; the publisher's write path was validated directly.

**One pre-existing, unrelated issue** surfaced during verification and was NOT
introduced here: `app/resilience/token_maintenance/schedule_registry.py` references
a missing setting `celery_token_maintenance_queue_name`, which blocks collection of
`test_queue_depth_publisher.py`. It lives outside this package and is left for a
separate fix. (A stale test fake in `test_circuit_state_probe.py` — also
pre-existing — was updated to match the real breaker interface so its two tests
pass.)
