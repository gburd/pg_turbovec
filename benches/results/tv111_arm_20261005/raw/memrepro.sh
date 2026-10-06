. ~/pgenv.sh
for arm in old new; do bash ~/swap.sh $arm >/dev/null; /mnt/nvme/venv/bin/python ~/memrepro.py $arm 30; done
bash ~/swap.sh new >/dev/null; echo MR_DONE
