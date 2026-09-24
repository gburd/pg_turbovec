# Z6 10M x 1024-d build measurement — instance record (written at launch)

Reproducing the EXACT config that OOM-killed before the per-tuple-context fix:
c7i.8xlarge (32 vCPU, 61 GiB), 10M x 1024-d, lists=3162.

- profile hotdog (585335547908), us-east-2b
- instance i-0b6954ee773696537 (c7i.8xlarge) ; tag run=z6ten-20260924-094517
- sg sg-02150d437a872566a (ssh 73.4.58.126/32 only) ; key ~/.ssh/z6ten-20260924-094517.pem
- 400 GB gp3, 12000 IOPS, 500 MB/s

## Teardown
```bash
aws ec2 terminate-instances   --profile hotdog --region us-east-2 --instance-ids i-0b6954ee773696537
aws ec2 delete-security-group --profile hotdog --region us-east-2 --group-id sg-02150d437a872566a
aws ec2 delete-key-pair       --profile hotdog --region us-east-2 --key-name z6ten-20260924-094517
```

SSH needs: -o IdentitiesOnly=yes -o IdentityAgent=none
Other tenants share this account; touch only run=z6ten-20260924-094517.
