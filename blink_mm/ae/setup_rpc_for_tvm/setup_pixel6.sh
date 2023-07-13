NUM_THREADS="${NUM_THREADS:=1}"

python ~/tvm-rpc/launch.py \
    --server localhost \
    --rpc-port 9001 \
    --tracker-port 9190 \
    --serial "1B261FDF6009KS" \
    --num-threads $NUM_THREADS \
    --taskset "C0" \
    --key pixel6