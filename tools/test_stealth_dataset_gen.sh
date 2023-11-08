for i in {1..10}
do
    python3 stealth_entry.py profiled_simple_simd_ops $i ../codelets/examples/genesys/configs/stealth_benchmark_16x16.json --num_points 5 --num_jobs 5
    if [ $? -ne 0 ]; then
        echo "Failed at $i"
        exit 1
    fi
    rm -rf stealth_output/compilation_output
done
echo "Success"
