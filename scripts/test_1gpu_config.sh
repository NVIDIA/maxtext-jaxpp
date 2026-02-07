set -e

if [ -n "$MODEL_CONFIG" ] && [ -n "$CONFIG_FILE" ]; then
    echo "Error: both MODEL_CONFIG and CONFIG_FILE are set"
    exit 1
fi

if [ -n "$CONFIG_FILE" ]; then
    source $CONFIG_FILE
fi

export N_PROCS=1
export N_GPUS=1

# Run plain JAX config
export JAXPP_CONFIG="
    max_target_length=64
"
bash ./scripts/run_local_mc.sh

# Run JaxPP config
export PARALLELISM_CONFIG="ici_pipeline_parallelism=1"

export JAXPP_CONFIG="
    scan_layers=False
    use_jaxpp=True
    schedule=interleaved_1f1b
    num_pipeline_microbatches=4
    num_pipeline_repeats=1
    max_target_length=64
"

bash ./scripts/run_local_mc.sh
