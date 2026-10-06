# Recheck-cost work (steps 1-7) — EC2 launch record (written at launch)
profile hotdog (account 170848442262), us-east-2, run=tvperf-20261006-135223
dev/test: i-02a585e79c18df3b8 (c7i.4xlarge, Sapphire Rapids AVX-512 VBMI+VNNI, Debian 13 amd64 ami-00ec124ea86e3fc0a, 200 GB gp3)
key /home/gburd/.ssh/tvperf-20261006-135223.pem ; sg sg-0403dbac5c85fd52b (ssh 73.4.58.126/32 only)
SSH: ssh -i /home/gburd/.ssh/tvperf-20261006-135223.pem -o IdentitiesOnly=yes -o IdentityAgent=none admin@18.219.111.242
Teardown: terminate every instance tagged run=tvperf-20261006-135223, then delete sg sg-0403dbac5c85fd52b and key pair tvperf-20261006-135223.
NOTE: untagged instances in this account are not ours.
