# Orphaned team-resource sweep

Deleting a team reclaims its event-store documents and nothing else. Its
Weaviate vectors, its Docker sandbox container and its workspace directory
survive it, indefinitely. This job finds those and removes them.

```bash
python -m akgentic.infra.maintenance                     # dry run — prints the plan
python -m akgentic.infra.maintenance --apply             # actually deletes
python -m akgentic.infra.maintenance --only docker       # one backend
```

Design rationale: **ADR-042**, in the workspace repo at
`_bmad-output/akgentic-infra/decisions/adr-042-orphaned-team-resource-sweep.md`
(this package is a submodule, so the file is not reachable by a relative link).

## What leaks, and why

`TeamManager.delete_team` calls `event_store.delete_team()` and deregisters the
team. Two other systems hold team-keyed state and hear nothing:

| Backend | What is left | How it is keyed |
|---|---|---|
| Weaviate | every vector object the team ingested | a `team_id` property on each object — collections are shared by all teams, so this is the *only* thing that says who owns an object |
| Docker | one stopped container per team | the container name, `sandbox-<team_id>`. `DockerSandboxActor` stops it on teardown and deliberately never runs `docker rm` |
| Workspace filesystem | the directory tree and its git journal | `$AKGENTIC_WORKSPACES_ROOT/<workspace_id or team_id>`, journalling to a sibling `<name>.git` |

**The workspace reaper deletes data, not runtime.** A vector can be re-ingested
and a container rebuilt from its image; the files an agent wrote cannot be
recovered from anywhere. Read its dry run before applying it, and if you want
only the recoverable half unattended on a schedule, run
`--only weaviate --only docker`.

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
and all 39 sandboxes looked orphaned. The guard is what stands between that
and a mass deletion. `--force` overrides it, for an operator who has read the
dry run.

**The grace period.** `--grace-seconds` (default 3600) holds back resources
younger than the threshold regardless of the live set, covering the window
where a container starts before its team's first persisted checkpoint. It
applies only where the backend exposes a creation time — Docker does, Weaviate
objects do not, and they appear only on ingest, long after the team is durable.

**Claims, not just team ids.** A workspace directory is named
`workspace_id or team_id`, and `workspace_id` is an operator-chosen override
that two teams may share — `WorkspaceTool(workspace_id="shared")` is a
supported configuration. A live team therefore routinely owns a directory
whose name is not its id, so every `workspace_id` on a card a live team
references is protected too. If a card cannot be resolved, that set is
incomplete and the sweep refuses rather than act on a partial answer.

**Never reaped:** running containers; symlinks; any workspace whose name is
not a UUID (a named shared tree belongs to whoever configured it, and no
team's deletion can orphan it); and anything unattributable — a container
whose name suffix is not a UUID, an object with no `team_id`.

## Configuration

Backends are discovered from the environment. Each is optional; an unconfigured
one is skipped, and an unreachable one is reported as *unavailable* rather than
as "no orphans".

| Variable | Effect |
|---|---|
| `AKGENTIC_WEAVIATE_URL` | enables the Weaviate reaper |
| `AKGENTIC_WEAVIATE_API_KEY` | optional cluster credential |
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
      - weaviate
      - --only
      - docker
```

The `--only` pair is deliberate: it keeps the two recoverable backends on the
schedule and leaves workspace deletion to a human who has read the plan. Drop
it once you are satisfied with what the workspace half proposes.

The Docker reaper shells out to the `docker` CLI, so a containerised sweep
needs the daemon socket mounted. Where that is unacceptable, run
`--only weaviate` in-cluster and the Docker half on the host.

## Extending it

A reaper is anything satisfying `TeamResourceReaper` — `scan()` returning
`ResourceRef`s tagged with their owning team, `purge(ref)`, `close()`. It
knows nothing about live teams; that decision belongs to the driver, which is
the only place the ordering above can be enforced once. Adding a backend means
adding a class and a line in `_build_reapers`.
