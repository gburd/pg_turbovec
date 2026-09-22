# Z5 delta validation — EC2 instance record

Written at launch so cleanup is possible even if this session dies.

- profile: hotdog (account 585335547908), region us-east-2
- instance: i-0c482726a6ca1ac87 (m7i.4xlarge, AVX-512 — latency-publishable)
- tag: run=z5delta-20260922-110327
- security group: sg-0602c58c9c3a642ec (ssh restricted to 73.4.58.126/32)
- key: ~/.ssh/z5delta-20260922-110327.pem  (also key-pair name z5delta-20260922-110327)

## Teardown (run all three)
```bash
aws ec2 terminate-instances --profile hotdog --region us-east-2 --instance-ids i-0c482726a6ca1ac87
aws ec2 delete-security-group --profile hotdog --region us-east-2 --group-id sg-0602c58c9c3a642ec   # after termination
aws ec2 delete-key-pair      --profile hotdog --region us-east-2 --key-name z5delta-20260922-110327
```

NOTE: five untagged instances were already running in this account at launch
(i-094404d62fe47d9ef, i-01dc5f18dcfe847f4, i-079fad872e65b6da7,
i-0e3ee55601154755e, i-06c848284a471f242). They are NOT ours — do not touch.

## Gotcha (cost me ~25 min)
SSH failed with "Authentications that can continue: publickey" on TWO
instances. Not AWS: my ssh AGENT offered its own keys and exhausted the
attempt limit before reaching `-i`. Always use:
  `-o IdentitiesOnly=yes -o IdentityAgent=none`
The first instance (i-0c482726a6ca1ac87) was terminated while chasing this;
capacity for m7i then vanished region-wide, so this run is c7i.4xlarge.
