set -x

shopt -s expand_aliases

alias l.='ls -d .* --color=auto'
alias ll='ls -l --color=auto'
alias ls='ls --color=auto'

# export HIP_VISIBLE_DEVICES=5
export HIP_VISIBLE_DEVICES=6

rocm-smi | egrep "$HIP_VISIBLE_DEVICES    |Device"
pip show triton
rocprofv3 --version
# triton_cache_dir=~/.triton/cache
# triton_cache_dir=/mnt/raid0/heyanguang/code/fa_triton/aiter/triton_cache1
triton_cache_dir=/mnt/raid0/heyanguang/code/fa_triton/aiter/triton_cache2
export TRITON_CACHE_DIR=${triton_cache_dir}

rm -rf ${triton_cache_dir}
export AITER_LOG_MORE=1




python ./amdgcn_edit/test_amdgcn_cfg.py > logx.test 2>&1
python ./amdgcn_edit/test_amdgcn_ddg.py >> logx.test 2>&1
python ./amdgcn_edit/test_amdgcn_latency.py >> logx.test 2>&1
python ./amdgcn_edit/test_amdgcn_passes.py >> logx.test 2>&1
python ./amdgcn_edit/test_amdgcn_verify.py >> logx.test 2>&1
python ./amdgcn_edit/test_amdgcn_register_slice.py >> logx.test 2>&1
cat logx.test | egrep "ALL TESTS PASSED"
cat logx.test | egrep "ALL TESTS PASSED" | wc -l
rm -f logx.test

python ./amdgcn_edit/amdgcn_ddg.py ./amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.amdgcn -o ./amdgcn_edit/ddg_trans_out_v1 --stats --save-json
python ./amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -t ./amdgcn_edit/trans_pass_list.json -r ./amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn
export TRITON_OVERRIDE_AMDGCN_FILE=/mnt/raid0/heyanguang/code/fa_triton/aiter/amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn
python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 80 -q 1 -c 2048 -n 16,1 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 1024

python ./amdgcn_edit/amdgcn_ddg.py ./amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.ps_256.amdgcn -o ./amdgcn_edit/ddg_trans_out_v1 --stats --save-json
python ./amdgcn_edit/amdgcn_ddg.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json -t ./amdgcn_edit/trans_pass_list.ps_256.json -r ./amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn
export TRITON_OVERRIDE_AMDGCN_FILE=/mnt/raid0/heyanguang/code/fa_triton/aiter/amdgcn_edit/ddg_trans_out_v1/modi_v2.amdgcn
python ./op_tests/triton_tests/test_pa_decode_gluon.py -b 80 -q 1 -c 2048 -n 16,1 -d 128 --block_size 64 --compute_type bf16 --quant_mode per_tensor --trans_v false --kv_varlen false --use_aot_impl false --quant_q_and_kv 0,0 --context_partition_size 256

python ./amdgcn_edit/amdgcn_ddg.py ./amdgcn_edit/pa_dot_kernel.no_iglp.bf16.kv_blk_64.kv_cmput_blk_128.amdgcn -o ./amdgcn_edit/ddg_trans_out_v1 --stats --save-json
python ./amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list ./amdgcn_edit/trans_pass_list.json --output-dir ./amdgcn_edit/debug_test --detail-pass 1
python ./amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list ./amdgcn_edit/trans_pass_list.json --output-dir ./amdgcn_edit/debug_test --detail-pass 3
python ./amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list ./amdgcn_edit/trans_pass_list.json --output-dir ./amdgcn_edit/debug_test --detail-pass 1 --detail-step 7 --start-move 0
python ./amdgcn_edit/debug_distribute_pass.py --load-json ./amdgcn_edit/ddg_trans_out_v1/analysis.json --pass-list ./amdgcn_edit/trans_pass_list.json --output-dir ./amdgcn_edit/debug_test --detail-pass 3 --detail-step 7 --start-move 0


set +x
