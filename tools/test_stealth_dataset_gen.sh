rm -rf stealth_output/compilation_output
for i in {0..10}
do
    python3 stealth_entry.py profiled_simple_simd_ops $i ../codelets/examples/genesys/configs/stealth_benchmark_16x16.json --num_points 3 --num_jobs 3
    if [ $? -ne 0 ]; then
        echo "Failed at $i"
        exit 1
    fi
    rm -rf stealth_output/compilation_output
done
echo "Success"
