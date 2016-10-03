This folder contains testcases for the project

test scripts for s3:

`test.sh`

```bash
for r in {0..10}; do
    file=data/${RANDOM}
    start=`date +'%s.%N'`
    ./filesys_test cat s3://dmlc/ilsvrc12/val.rec >$file
    # ./filesys_test cat s3://dmlc/cifar10/train.rec >$file
    end=`date +'%s.%N'`
    res=$(echo "$end - $start" | bc -l)
    md5=`md5sum $file`
    rm $file
    echo "job $1, rp $r, $md5, time $res"
done
echo "job $1 done"
```

`run.sh`

```bash
mkdir -p data
rm -f data/*
for i in {0..9}; do
    bash test.sh $i &
    sleep 1
done
wait
```
