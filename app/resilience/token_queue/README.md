# Token Queue — a beginner's guide

> **Read this first.** This is **Layer 4: queue absorption** — the piece that makes
> the `/acquire` fast path possible by moving the *durable database write* off the
> request and onto a RabbitMQ queue. It looks intimidating because it implements a
> full **retry-with-backoff → dead-letter** pipeline using RabbitMQ's own
> mechanics. This guide explains the *why*, the *flow*, and the clever bits (the
> TTL-based delayed retry especially), from scratch.

---

## 1. The job in one sentence

When `/acquire` reserves tokens in Redis (~1ms), it still has to write that
reservation **durably** to PostgreSQL. Doing that write *on the request* would
throw away the speed. So this module **hands the write off to RabbitMQ** and
returns immediately; a separate consumer process does the actual database write in
the background — reliably, with retries, and with a dead-letter safety net.

The reason it must be a real queue (and not a simple background task) is
**durability across a crash**: once the message is in RabbitMQ, it survives an API
restart and still gets written. See the main resilience discussion for that
contrast with the (removed) Celery scheduler.

---

## 2. The mental model: a durable outbox

Think of it as an **outbox**. The API writes a note ("persist this allocation")
into a durable mailbox and walks away. A dedicated mail carrier (the consumer)
picks up notes and files them in the database. If filing fails, the note goes into
a "try again in 30s" tray; if it keeps failing, it goes into a "needs a human"
tray (the DLQ). No note is ever silently lost.

The whole module is the machinery for that outbox: **who writes notes**
(publisher), **the trays and how they're wired** (topology), **who reads and files
them** (consumer), and **what filing actually does** (handlers).

---

## 3. The happy path (one allocation, start to finish)

```
/acquire  ──reserve in Redis (Layer 3)──►  publisher.publish_allocation_request(payload)
                                                     │  (publish to the WORK queue)
                                                     ▼
                                     RabbitMQ:  token.allocation.work  (durable, quorum)
                                                     │
                                consumer._on_work_message(body)  ◄── the consumer process
                                                     │
                                   persist_allocation_message(body) → write to PostgreSQL
                                                     │
                                              message.ack()   ✔ done, removed from the queue
```

`ack()` is the key word: RabbitMQ only drops a message once the consumer
**acknowledges** it succeeded. If the consumer crashes mid-write without acking,
RabbitMQ redelivers the message to another consumer. That's the at-least-once
guarantee that makes this durable.

---

## 4. The unhappy path: retry, then dead-letter

What if the PostgreSQL write fails (DB blip, timeout)? The consumer doesn't just
drop it or spin on it. It walks a ladder:

```
persist fails (attempt N)
      │
      ├─ attempts left?  ── yes ──►  publisher.publish_retry_request(attempt=N+1)
      │                                   │ (publish to a delayed RETRY queue)
      │                                   ▼
      │                        token.allocation.retry.30s   ← waits ~30s, then comes back
      │                                   │
      │                                   └──► (delay elapses) message re-appears on the WORK queue
      │                                        → consumer tries again
      │
      └─ attempts exhausted ──►  publisher.publish_dlq_notification(reason)
                                        │ (publish to the DEAD-LETTER queue)
                                        ▼
                              token.allocation.dlq   → consumer._on_dlq_message → alert + undo
```

### The clever bit: delayed retry with *no scheduler* (TTL + dead-letter)

How does "wait 30 seconds, then retry" work without any timer in our code? Pure
RabbitMQ mechanics, and it's worth understanding because it's a classic pattern:

- Each **retry queue** (`…retry.30s`, `…retry.120s`, …) is declared with two magic
  arguments in [topology.py](./topology.py):
  - `x-message-ttl = 30000` — a message **expires** after 30 seconds.
  - `x-dead-letter-exchange` / `x-dead-letter-routing-key` pointing **back at the
    work queue.**
- The retry queue has **no consumer.** A message published there just *sits* there.
- When its TTL expires, RabbitMQ automatically **dead-letters** it — which, because
  of those arguments, routes it **back onto the work queue** for another attempt.

So the "delay" is just a message expiring in a parking-lot queue, and the "retry"
is RabbitMQ's dead-letter routing. We get scheduled retries for free from the
broker — no cron, no scheduler, no `sleep`. The retry *schedule* (how many stages
and their delays) comes from `settings.token_queue_retry_schedule_seconds`, one
retry queue per stage.

### The dead-letter queue (DLQ) — the "needs a human" tray

After the configured retries are exhausted, the message is published to the **DLQ**.
Its consumer callback (`_on_dlq_message` → `process_dlq_alert` in
[handlers.py](./handlers.py)) does two important things:

1. **Undo the Redis reservation.** The allocation failed to persist for good, so the
   tokens reserved back in Layer 3 must be *released* — otherwise Redis would keep
   counting capacity that was never actually used (a leak). It calls
   `release_tokens(...)` on the redis_token_counter service.
2. **Alert loudly** (`logger.critical`) with the full payload for manual review.

This closes the loop: even a terminal failure leaves the system *consistent* (no
leaked capacity) and *observable* (a human is told).

---

## 5. The topology, in plain terms ([topology.py](./topology.py))

RabbitMQ routing has three nouns; here's the map:

- **Exchanges** (the "routers"): `TOKEN_EXCHANGE` routes normal traffic;
  `TOKEN_DLX` (dead-letter exchange) routes terminal failures.
- **Queues** (the "mailboxes"): the **work** queue, one **retry** queue per delay
  stage, and the **DLQ**. All are `x-queue-type: quorum` — replicated across nodes
  for high availability, so a broker node dying doesn't lose messages.
- **Routing keys** (the "addresses"): a message's routing key + the exchange decide
  which queue it lands in (`token.allocate`, `token.allocate.retry.30s`,
  `token.allocate.dead`).

`declare_token_queues()` creates all of this at app startup, **exchanges before
queues** (a queue can't bind to an exchange that doesn't exist yet — that ordering
is even asserted in the tests).

Two headers ride along on every message and carry the retry state:
`x-token-retry-attempt` (which attempt this is) and `x-token-retry-reason` (why the
last attempt failed).

---

## 6. The pieces, file by file

- [`topology.py`](./topology.py) — **the map.** Exchanges, work/retry/DLQ queues,
  their TTL+dead-letter wiring, and `declare_token_queues()`.
- [`publisher.py`](./publisher.py) — **writes notes.** `TokenAllocationPublisher`
  publishes work / retry / DLQ messages, each wrapped in the **RabbitMQ circuit
  breaker**; if the breaker is OPEN, the caller falls back to a synchronous DB write.
- [`consumer.py`](./consumer.py) — **reads and files notes.** The long-running
  `ConsumerMixin` loop; `_on_work_message` is the retry/DLQ decision tree above.
  Runs as its own process (or a pool of them), separate from the API.
- [`handlers.py`](./handlers.py) — **what filing does.** `persist_allocation_message`
  (write to PostgreSQL) and `process_dlq_alert` (release Redis tokens + alert).
- [`healthcheck.py`](./healthcheck.py) — a shell-safe broker-connectivity probe for
  the consumer container.

---

## 7. Two safety mechanisms you'll see throughout

- **The RabbitMQ circuit breaker.** Every publish goes through the `rmq` breaker
  (the same pattern as the `circuit_breaker` module). If RabbitMQ is down, publishes
  fail fast and the caller uses the **synchronous DB path** instead — the fast path
  degrades, it never hard-fails.
- **Backoff-requeue.** If the consumer *can't even publish* a retry/DLQ message
  because the breaker is open, it doesn't drop the message — it sleeps briefly and
  `reject(requeue=True)`s it so the broker will redeliver it later
  (`_backoff_requeue`). Nothing is lost even when the broker is flaky.

---

## 8. Can I reuse this? (production perspective)

Yes — this is a **reference implementation of durable async work handoff**, and the
patterns are broadly reusable:

- **The outbox/handoff shape** (publish on the hot path, persist in a consumer) is
  how you decouple a fast API from a slower write without losing durability.
- **TTL + dead-letter delayed retry** is a standard RabbitMQ idiom for "retry later
  without a scheduler" — reuse it anywhere you need backoff.
- **Retry-ladder → DLQ → compensating action** (here: release the Redis reservation)
  is the correct way to keep a system consistent when async work terminally fails.

Seams to swap when porting: the payload contracts (`resilience_models`), the
routing/queue names and retry schedule (all in `settings`), and the "what to do on
success/terminal-failure" side effects in `handlers.py`. The broker wiring itself
is generic.

---

## 9. Thirty-second recap

- This is the **durable outbox** that lets `/acquire` be fast: publish the DB write
  to RabbitMQ, return, and let a consumer persist it in the background.
- **Happy path:** publish → work queue → consumer writes to Postgres → `ack`.
- **Failure:** retry via **TTL parking-lot queues that dead-letter back to work**
  (delayed retry with no scheduler) → after N attempts → **DLQ** → release the Redis
  reservation + alert a human.
- Everything is **at-least-once + durable** (quorum queues, ack-on-success,
  requeue-on-trouble), and every publish is guarded by the **RabbitMQ breaker** with
  a synchronous DB fallback.
- Unlike the Celery scheduler we removed, this queue is **load-bearing** — it's the
  backbone of the fast-path architecture.
