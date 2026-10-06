# turbovec 1.1.1 qualification — EC2 launch record (written at launch)
profile hotdog (account 170848442262), us-east-2
run=tv111-20261005-164703 ; instance i-0941cac138bba5c33 (c8gd.8xlarge, Graviton4 / Neoverse V2, Debian 13 arm64 ami-05b1ff755a125cbf9)
key /home/gburd/.ssh/tv111-20261005-164703.pem ; sg sg-07e3557ff5cdf722c (ssh 73.4.58.126/32 only)
SSH: ssh -i /home/gburd/.ssh/tv111-20261005-164703.pem -o IdentitiesOnly=yes -o IdentityAgent=none admin@3.137.155.155
Teardown:
  aws ec2 terminate-instances   --profile hotdog --region us-east-2 --instance-ids i-0941cac138bba5c33
  aws ec2 delete-security-group --profile hotdog --region us-east-2 --group-id sg-07e3557ff5cdf722c
  aws ec2 delete-key-pair       --profile hotdog --region us-east-2 --key-name tv111-20261005-164703
NOTE: other running instances in this account are NOT ours (no run= tag). Do not touch.
second instance (soak + aarch64 pg_test): i-06c26bd9a1c103737 (c8gd.4xlarge). Teardown: add it to terminate-instances.

TEARDOWN DONE 2026-10-06T04:57:03Z: both instances terminated (verified), sg-07e3557ff5cdf722c and key pair deleted, no tagged volumes remain.
