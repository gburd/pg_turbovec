# Z5 >RAM validation — EC2 instance record (written at launch)

- profile hotdog (585335547908), region us-east-2c
- instance i-02a0cca89da185af0 (c7i.8xlarge, 32 vCPU, AVX-512)
- tag run=z5ram-20260922-134845 ; sg sg-0d743f14237a7ea2f (ssh 73.4.58.126/32 only) ; key ~/.ssh/z5ram-20260922-134845.pem
- 400 GB gp3, 12000 IOPS, 500 MB/s

## Teardown
```bash
aws ec2 terminate-instances   --profile hotdog --region us-east-2 --instance-ids i-02a0cca89da185af0
aws ec2 delete-security-group --profile hotdog --region us-east-2 --group-id sg-0d743f14237a7ea2f
aws ec2 delete-key-pair       --profile hotdog --region us-east-2 --key-name z5ram-20260922-134845
```

## SSH (agent MUST be bypassed — see z5_delta_20260922/INSTANCE.md)
```bash
ssh -o IdentitiesOnly=yes -o IdentityAgent=none -i ~/.ssh/z5ram-20260922-134845.pem ubuntu@<ip>
```
Other users' untagged instances share this account; touch only run=z5ram-20260922-134845.
