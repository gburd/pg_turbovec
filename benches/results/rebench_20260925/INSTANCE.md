# IVF-vs-HNSW end-to-end re-bench — instance (launch record)
account 373102893032 (hotdog), us-east-2 ; key ~/.ssh/rebench-20260925-141410.pem ; sg sg-0e19ce4d71b975a7e (ssh 73.4.58.126/32)
instance i-0aef2fe88a6f01b38 (c7i.8xlarge) ; run=rebench-20260925-141410
SSH: -o IdentitiesOnly=yes -o IdentityAgent=none
Teardown: terminate-instances i-0aef2fe88a6f01b38 ; delete-security-group sg-0e19ce4d71b975a7e ; delete-key-pair rebench-20260925-141410
