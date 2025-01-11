from nemo.collections import llm
import nemo_run as run

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=llm.llama31_8b.model(),
        source="hf://meta-llama/Meta-Llama-3.1-8B",
        overwrite=False,
    )


def configure_finetuning_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = llm.llama31_8b.finetune_recipe(
        name="llama3_lora",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    recipe.trainer.max_steps = 100
    recipe.trainer.num_sanity_val_steps = 0

    # Async checkpointing doesn't work with PEFT
    recipe.trainer.strategy.ckpt_async_save = False

    # Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.val_check_interval = 100

    # This is currently required for LoRA/PEFT
    recipe.trainer.strategy.ddp = "megatron"
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def run_finetuning():
    import_ckpt = configure_checkpoint_conversion()
    finetune = configure_finetuning_recipe(nodes=1, gpus_per_node=1)
    #from nemo.collections.llm.recipes.llama32_1b import finetune_recipe
    #finetune = finetune_recipe(name="llama32_1b_finetune", num_nodes=1, num_gpus_per_node=1)
    executor = local_executor_torchrun(nodes=finetune.trainer.num_nodes, devices=finetune.trainer.devices)
    executor.env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    with run.Experiment("llama3-8b-peft-finetuning") as exp:
        exp.add(
            import_ckpt, executor=run.LocalExecutor(), name="import_from_hf"
        )  # We don't need torchrun for the checkpoint conversion
        exp.add(finetune, executor=executor, name="peft_finetuning")
        exp.run(sequential=True, tail_logs=True)  # This will run the tasks sequentially and stream the logs


# Wrap the call in an if __name__ == "__main__": block to work with Python's multiprocessing module.
if __name__ == "__main__":
    run_finetuning()
