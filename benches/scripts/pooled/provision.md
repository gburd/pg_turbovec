# Provisioning the three-tier pooled benchmark rig on EC2

> This is the **real headline rig** from `benches/METHODOLOGY.md`: PostgreSQL +
> pg_turbovec on one **bare-metal** on-demand instance, a connection pooler on
> its **own** on-demand instance, and **N spot** load-generator instances —
> all in one VPC/subnet. A collapsed single-host run (DB + pooler + driver on
> one box, e.g. the S0 smoke) is a **diagnostic**, never a headline.
>
> This file documents how to launch it. The scale-plan S0 deliverable did NOT
> launch it (budget + user approval gate); the scripts (`setup_db.sh`,
> `setup_pooler.sh`, `run_load.sh`, `aggregate.py`, `fresh_backend_cost.sh`)
> are tier-agnostic and run unchanged once the hosts exist.

## Topology

```
  [ load fleet: N spot ]        [ pooler: 1 on-demand ]      [ DB: 1 on-demand METAL ]
  m7i/c7i.large × N        →    c7i.2xlarge (PgBouncer)  →    i7i.metal-24xl / r8gd.metal
  run_load.sh per node          setup_pooler.sh               setup_db.sh
  (pgbench, open+closed)        transaction pooling           PG18 + pg_turbovec, tuned
```

- **DB on metal, on-demand.** Metal so NUMA, huge pages, and local-NVMe ZFS
  are real. On-demand because a spot reclaim mid-run **voids the result**.
- **Pooler on its own on-demand instance.** Its CPU + connection multiplexing
  are measured, not stolen from DB or drivers. The DB sees exactly
  `default_pool_size` backends.
- **Load fleet = many small SPOT instances.** Stateless, restartable; spot is
  fine. Enough nodes that no single driver is CPU/NIC-bound (verify each node
  < saturation, or you measure the driver, not the server).

## AWS burner rules (from AGENTS.md § AWS burner accounts) — MANDATORY

1. **Verify the profile first:** `aws sts get-caller-identity --profile <burner>`.
   `InvalidClientTokenId` = the account expired; do not retry, escalate.
2. **Never open an SG to `0.0.0.0/0`.** SSH scoped to the operator /32
   (`curl -s https://checkip.amazonaws.com`), or SSM with no inbound port.
   Intra-rig traffic (pooler↔DB 28818, fleet↔pooler 6432) scoped to the rig SG.
3. **Tag every instance `run=<id>`** and touch only your own tag — these
   accounts are shared.
4. **Pull artefacts continuously** (`scp` the per-driver JSONs as each sweep
   finishes), not at the end.
5. **Terminate-by-own-tag first, report second.** Write the teardown script at
   launch time.

## Launch sketch (parameterise; do not hardcode)

```bash
PROFILE=<burner>; REGION=us-east-2; RUN=s1-pooled-$(date +%Y%m%d)
MYIP=$(curl -s https://checkip.amazonaws.com)/32
aws sts get-caller-identity --profile "$PROFILE"      # GATE: must succeed

# one VPC/subnet already exists or create one; then a rig SG:
SG=$(aws ec2 create-security-group --profile $PROFILE --region $REGION \
      --group-name rig-$RUN --description "pooled rig $RUN" \
      --vpc-id $VPC --query GroupId --output text)
# SSH from operator only; intra-rig on the pooler/DB ports from within the SG:
aws ec2 authorize-security-group-ingress --profile $PROFILE --region $REGION \
      --group-id $SG --protocol tcp --port 22 --cidr $MYIP
aws ec2 authorize-security-group-ingress --profile $PROFILE --region $REGION \
      --group-id $SG --protocol tcp --port 28818 --source-group $SG   # DB
aws ec2 authorize-security-group-ingress --profile $PROFILE --region $REGION \
      --group-id $SG --protocol tcp --port 6432  --source-group $SG   # pooler

# DB tier — on-demand metal:
aws ec2 run-instances --profile $PROFILE --region $REGION \
  --image-id <debian-ami> --instance-type i7i.metal-24xl \
  --key-name <key> --security-group-ids $SG --subnet-id $SUBNET \
  --tag-specifications "ResourceType=instance,Tags=[{Key=run,Value=$RUN},{Key=tier,Value=db}]"

# Pooler tier — on-demand:
aws ec2 run-instances --profile $PROFILE --region $REGION \
  --image-id <debian-ami> --instance-type c7i.2xlarge \
  --key-name <key> --security-group-ids $SG --subnet-id $SUBNET \
  --tag-specifications "ResourceType=instance,Tags=[{Key=run,Value=$RUN},{Key=tier,Value=pooler}]"

# Load fleet — N spot:
aws ec2 run-instances --profile $PROFILE --region $REGION \
  --image-id <debian-ami> --instance-type c7i.large --count N \
  --instance-market-options 'MarketType=spot' \
  --key-name <key> --security-group-ids $SG --subnet-id $SUBNET \
  --tag-specifications "ResourceType=instance,Tags=[{Key=run,Value=$RUN},{Key=tier,Value=load}]"
```

## Storage on the DB metal host (METHODOLOGY.md target)

ZFS on local NVMe with a separate ZIL (SLOG) + L2ARC. Record
`recordsize`/`compression`/`atime`/`logbias`, ARC size, and the ZIL/L2ARC
devices. A plain XFS/ext4-on-NVMe run is a *substrate* row, labelled so.
Reserve huge pages before starting PG (`setup_db.sh` prints the sysctl).

## Orchestration order

1. All tiers up; record `client↔pooler↔db` RTTs (`ping`, and a trivial
   `SELECT 1` latency through each hop).
2. On DB: `DB_PORT=28818 CORPUS_ROWS=<big> DIM=768 INDEX_KIND=ivf LISTS=<N> \
   SHARED_BUFFERS=<tens of GB> bash setup_db.sh`, then restart PG for
   shared_buffers/huge_pages. **Corpus must exceed shared_buffers + page cache**
   so the storage path is exercised (METHODOLOGY.md).
3. On pooler: `DB_HOST=<db-ip> POOLER_PORT=6432 DEFAULT_POOL_SIZE=<sustainable
   backend count> MAX_CLIENT_CONN=<thousands> bash setup_pooler.sh start`.
4. On each load node: `POOLER_HOST=<pooler-ip> DRIVER_ID=node<k> \
   CLIENTS="..." OPEN_RATES="..." SCAN_PARALLELISM=1 bash run_load.sh`
   (staggered start; synchronised clocks via chrony; discard warmup). Repeat
   with `SCAN_PARALLELISM=auto` for the equal-CPU pair.
5. `fresh_backend_cost.sh` on the DB host (direct) for the fresh-backend row.
6. `scp` every `<run_tag>_<driver_id>.json` back; `aggregate.py OUT.md *.json`.

## Teardown (by tag, always)

```bash
aws ec2 describe-instances --profile $PROFILE --region $REGION \
  --filters "Name=tag:run,Values=$RUN" \
  --query 'Reservations[].Instances[].InstanceId' --output text \
| xargs -r aws ec2 terminate-instances --profile $PROFILE --region $REGION --instance-ids
aws ec2 delete-security-group --profile $PROFILE --region $REGION --group-id $SG
```
