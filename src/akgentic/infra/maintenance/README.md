# Orphaned team-resource sweep

Deleting a team reclaims its event-store documents and nothing else. Its
vector-store rows and its workspace directory survive it, indefinitely. This job
finds those and removes them.

```bash
python -m akgentic.infra.maintenance                     # dry run — prints the plan
python -m akgentic.infra.maintenance --apply             # actually deletes
python -m akgentic.infra.maintenance --only vector       # one kind of backend
```

Design rationale: **ADR-042**, in the workspace repo at
`_bmad-output/akgentic-infra/decisions/adr-042-orphaned-team-resource-sweep.md`
(this package is a submodule, so the file is not reachable by a relative link).

## What leaks, and why

`TeamManager.delete_team` calls `event_store.delete_team()` and deregisters the
team. Two other kinds of system hold team-keyed state and hear nothing:

| Backend | What is left | How it is keyed |
|---|---|---|
| Weaviate | every vector object the team ingested | a `team_id` property on each object — collections are shared by all teams, so this is the *only* thing that says who owns an object |
| Qdrant | every point the team ingested | a `team_id` payload key on each point — the same mechanism, same leak |
| Workspace filesystem | the directory tree and its git journal | `$AKGENTIC_WORKSPACES_ROOT/<workspace_id or team_id>`, journalling to a sibling `<name>.git` |

The vector reapers are resolved through `akgentic-tool`'s backend registry, one
per backend a deployment has provisioned — so a cluster running only Qdrant is
swept rather than reported clean. Backends that hold nothing reclaimable (the
in-memory index, the local on-disk index) are declared as such in
`vector_backends.py`, and a test fails if a newly registered backend is neither
reapable nor excused.

**There is no sandbox-container reaper.** The sandbox names its container after
random hex rather than after a team, and removes it on every stop, so a reaper
could attribute nothing and would report every deployment clean — worse than
none at all.

**The workspace reaper deletes data, not runtime.** A vector row can be
re-ingested from the source it came from; the files an agent wrote cannot be
recovered from anywhere. Read its dry run before applying it, and if you want
only the recoverable half unattended on a schedule, run `--only vector`.

## How it works

Three steps, in this order:

1. **Scan** every backend for team-owned resources.
2. **Read** the protected set from the `EventStore` — every team present and
   not `DELETED`, plus every `workspace_id` those teams' agent cards declare.
   A `STOPPED` team is resumable and keeps everything.
3. **Purge** the resources nothing in that set claims.

**Step 1 must precede step 2.** A team created while the sweep runs is absent
from the scan snapshot, so it cannot be reaped whatever the live set says.
Reversed, the same team would be missing from the live set and present in the
scan — and the sweep would delete a team created moments ago. There is no lock
here; the ordering is what makes the race benign, and a test asserts it.

## Safety rails

**Dry run is the default.** Without `--apply` nothing is deleted.

**The blast-radius guard.** The sweep refuses to apply a plan that is
implausibly large, exits `3`, and deletes nothing:

- the live team set is **empty** while resources are condemned, or
- more than `--max-orphan-fraction` (default `0.5`) of the scan is condemned.

This exists because the event store answers "which teams exist" on a
best-effort basis — the YAML and Mongo backends both log and skip a document
they cannot parse. On the first machine this ran on, every stored team failed
`Process` validation after a schema change, so `list_teams()` returned nothing
and all 39 live teams' resources looked orphaned. The guard is what stands
between that and a mass deletion. `--force` overrides it, for an operator who has read the
dry run.

**The grace period.** `--grace-seconds` (default 3600) holds back resources
younger than the threshold regardless of the live set, covering the window where
a resource is created before its team's first persisted checkpoint. It applies
only where the backend exposes an age — the workspace reaper does, by directory
mtime, so an actively written tree reads as young and survives a sweep that
misjudged it. Vector rows do not: they appear only on ingest, long after the
team is durable, and expose no cheap creation time per `(collection, team)`.

**Claims, not just team ids.** A workspace directory is named
`workspace_id or team_id`, and `workspace_id` is an operator-chosen override
that two teams may share — `WorkspaceTool(workspace_id="shared")` is a
supported configuration. A live team therefore routinely owns a directory
whose name is not its id, so every `workspace_id` on a card a live team
references is protected too. If a card cannot be resolved, that set is
incomplete and the sweep refuses rather than act on a partial answer.

**Never reaped:** symlinks; any workspace whose name is not a UUID (a named
shared tree belongs to whoever configured it, and no team's deletion can orphan
it); any collection that is not team-scoped, where `team_id` records who *wrote*
a row rather than who owns it, and which both backends refuse to delete from by
team anyway; and anything unattributable — a row with no `team_id`.

## Configuration

Backends are discovered from the environment. Each is optional; an unconfigured
one is skipped, and an unreachable one is reported as *unavailable* rather than
as "no orphans".

| Variable | Effect |
|---|---|
| `AKGENTIC_WEAVIATE_URL` | provisions Weaviate, and so its reaper |
| `AKGENTIC_WEAVIATE_API_KEY` | optional Weaviate credential |
| `AKGENTIC_QDRANT_URL` | provisions Qdrant, and so its reaper |
| `AKGENTIC_QDRANT_API_KEY` | optional Qdrant credential |
| `AKGENTIC_WORKSPACES_ROOT` | workspace root to sweep (default `./workspaces`) |
| `MONGO_URI` + `MONGO_DB` | read the live team set from Mongo (enterprise, department) |
| `AKGENTIC_EVENT_STORE_PATH` | filesystem event-store root (community) — used when the Mongo pair is not set |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | swept cleanly — dry run or applied, nothing failed |
| 1 | a backend was unreachable, or a purge failed |
| 2 | the live team set could not be read; nothing was scanned or deleted |
| 3 | the blast-radius guard refused the plan; nothing was deleted |

## Running it on a schedule

Daily is ample — the resources are inert, and the cost of reclaiming them a
day late is disk. Alert on exit `3` (needs a human) separately from exit `1`
(usually clears itself).

```yaml
# Kubernetes CronJob, abridged
schedule: "0 3 * * *"
containers:
  - name: sweep
    image: akgentic-infra:latest
    command:
      - python
      - -m
      - akgentic.infra.maintenance
      - --apply
      - --only
      - vector
```

The `--only vector` is deliberate: it keeps the recoverable backends on the
schedule — every vector store the deployment runs — and leaves workspace
deletion to a human who has read the plan. Drop it once you are satisfied with
what the workspace half proposes.

## Extending it

A reaper is anything satisfying `TeamResourceReaper` — `scan()` returning
`ResourceRef`s tagged with their owning team, `purge(ref)`, `close()`. It
knows nothing about live teams; that decision belongs to the driver, which is
the only place the ordering above can be enforced once. Adding a *kind* of
backend means adding a class and a line in `_build_reapers`; adding a *vector*
backend means one entry in `BACKEND_DISPOSITIONS`, naming either how to read its
team ids or why it holds nothing to reclaim.
