# Z6 build-memory profiling — instance record (written at launch)

- profile hotdog (585335547908), us-east-2b
- instance i-050657e511d9ecd3f (c7i.4xlarge) ; tag run=z6prof-20260922-165444
- sg sg-06398d106ad0929d0 (ssh 73.4.58.126/32) ; key ~/.ssh/z6prof-20260922-165444.pem

## Teardown
```bash
aws ec2 terminate-instances   --profile hotdog --region us-east-2 --instance-ids i-050657e511d9ecd3f
aws ec2 delete-security-group --profile hotdog --region us-east-2 --group-id sg-06398d106ad0929d0
aws ec2 delete-key-pair       --profile hotdog --region us-east-2 --key-name z6prof-20260922-165444
```
SSH needs: -o IdentitiesOnly=yes -o IdentityAgent=none
Other tenants' untagged instances share this account; touch only run=z6prof-20260922-165444.

## Second instance (cross-term test)
- i-0ad266bd77b62fb6f (c7i.4xlarge) us-east-2b, tag run=z6cross-20260922-225815, sg sg-0b8a750ab7c60e9dc, key ~/.ssh/z6cross-20260922-225815.pem
- teardown: terminate-instances i-0ad266bd77b62fb6f ; delete-security-group sg-0b8a750ab7c60e9dc ; delete-key-pair z6cross-20260922-225815

## Third instance (allocation tracing, session 4)
- i-06da21f8ae95bfa2a (c7i.4xlarge) us-east-2b, tag run=z6alloc-20260923-065610, sg sg-0c1614861fcc178b2, key ~/.ssh/z6alloc-20260923-065610.pem
- teardown: terminate-instances i-06da21f8ae95bfa2a ; delete-security-group sg-0c1614861fcc178b2 ; delete-key-pair z6alloc-20260923-065610

## Fourth instance (scan-phase measurement, session 5)
- i-08cfafdb63e6f39fb (c7i.4xlarge) us-east-2c, tag run=z6scan-20260923-150737, sg sg-0074f7a0420367500, key ~/.ssh/z6scan-20260923-150737.pem
- teardown: terminate-instances i-08cfafdb63e6f39fb ; delete-security-group sg-0074f7a0420367500 ; delete-key-pair z6scan-20260923-150737

## Fifth instance (fix verification, session 6)
- i-07841f9aad379271b (c7i.4xlarge) us-east-2b, tag run=z6fix-20260923-170407, sg sg-039887101fc6b1dba, key ~/.ssh/z6fix-20260923-170407.pem
- teardown: terminate-instances i-07841f9aad379271b ; delete-security-group sg-039887101fc6b1dba ; delete-key-pair z6fix-20260923-170407
