# --xla_dump_hlo_pass_re=.*
# export XLA_FLAGS="--xla_dump_hlo_as_html --xla_dump_hlo_as_text --xla_dump_to='./llama3-hlos-pp2' --xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_experimental_enable_fusion_autotuner=false --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_command_buffer=''"

source scripts/llama3.3_proxy_config.sh

export PARALLELISM_CONFIG="
    dcn_pipeline_parallelism=1 ici_pipeline_parallelism=2
    ici_data_parallelism=1
    ici_context_parallelism=2
    ici_tensor_parallelism=2
    ici_fsdp_parallelism=1
"

export JAXPP_CONFIG="
    scan_layers=False
    use_jaxpp=True
    schedule=interleaved_1f1b
    num_pipeline_microbatches=16
    num_pipeline_repeats=2
    profiler=xplane
    per_device_batch_size=16
    max_target_length=4096
"

export N_PROCS=8
export N_GPUS=1

bash ./scripts/run_local_mc.sh
