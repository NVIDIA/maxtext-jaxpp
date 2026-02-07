export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_experimental_enable_fusion_autotuner=false --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_command_buffer=''"

source scripts/gpt3_proxy_config.sh

export PARALLELISM_CONFIG="
    dcn_pipeline_parallelism=1 ici_pipeline_parallelism=4
    ici_data_parallelism=1
    ici_tensor_parallelism=2
    ici_expert_parallelism=1
    ici_fsdp_parallelism=1
"

export JAXPP_CONFIG="
    scan_layers=False
    use_jaxpp=True
    schedule=interleaved_1f1b
    num_pipeline_microbatches=16
    num_pipeline_repeats=2
    per_device_batch_size=16
    max_target_length=2048
"

export N_PROCS=8
export N_GPUS=1

bash ./scripts/run_local_mc.sh