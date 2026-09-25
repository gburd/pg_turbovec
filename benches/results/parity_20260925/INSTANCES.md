# Parity-gap work — EC2 instances (written at launch)

profile hotdog (account 373102893032), region us-east-2
key ~/.ssh/parity-20260925-094246.pem ; sg sg-069628f5a18f04d76 (ssh 73.4.58.126/32 only)
SSH MUST use: -o IdentitiesOnly=yes -o IdentityAgent=none  (agent-key trap, prior sessions)

- item1 IVF-vs-HNSW-at-recall : i-0c62e019c0fc5992f
- item2 cold-scan mmap        : i-0d6bb1d2aa48f5ce0

## Teardown (run for each IID, then the shared SG + key)
```bash
aws ec2 terminate-instances   --profile hotdog --region us-east-2 --instance-ids i-0c62e019c0fc5992f i-0d6bb1d2aa48f5ce0
aws ec2 delete-security-group --profile hotdog --region us-east-2 --group-id sg-069628f5a18f04d76
aws ec2 delete-key-pair        --profile hotdog --region us-east-2 --key-name parity-20260925-094246
```

NOTE: 6 untagged instances (run=None) belong to OTHER users of this account. Do NOT touch.
