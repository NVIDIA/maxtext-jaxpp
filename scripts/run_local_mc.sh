export TEST_CONFIG="
    override_model_config=True dataset_type=synthetic steps=10
"

export COMMAND="python3 -u -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    base_output_directory=run_local_mc_outputs \
    run_name=run_$(date +%Y-%m-%d-%H:%M:%S) \
    enable_checkpointing=false \
    async_checkpointing=false \
    dtype=bfloat16 \
    weight_dtype=bfloat16 \
    hardware=gpu \
    $MODEL_CONFIG \
    $TEST_CONFIG \
    $PARALLELISM_CONFIG
    $JAXPP_CONFIG"

bash ./scripts/local_mc.sh
