from nemo.collections import llm
import nemo_run as run

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model=llm.qwen2_7b.model(),
        source="hf://Qwen/Qwen2-7B",
        overwrite=False,
    )

from pytorch_lightning.loggers import WandbLogger
from nemo import lightning as nl

def configure_finetuning_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = llm.qwen2_7b.finetune_recipe(
        name="qwen2_lora",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )

    #recipe.log.wandb.project = "nemo"
    #recipe.log.wandb.entity = "pvduy"
    #recipe.log.wandb.name = "qwen2_lora"
    # add wandb follow this: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/exp-manager.html

    recipe.trainer.max_steps = 20000
    recipe.trainer.num_sanity_val_steps = 0
    
    # Async checkpointing doesn't work with PEFT
    recipe.trainer.strategy.ckpt_async_save = False

    # Need to set this to 1 since the default is 2
    recipe.trainer.strategy.context_parallel_size = 1
    recipe.trainer.val_check_interval = 1000

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
    executor = local_executor_torchrun(nodes=finetune.trainer.num_nodes, devices=finetune.trainer.devices)
    executor.env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    with run.Experiment("qwen2-7b-peft-finetuning") as exp:
        exp.add(
            import_ckpt, executor=run.LocalExecutor(), name="import_from_hf"
        )  # We don't need torchrun for the checkpoint conversion
        exp.add(finetune, executor=executor, name="peft_finetuning")
        exp.run(sequential=True, tail_logs=True)  # This will run the tasks sequentially and stream the logs


# Wrap the call in an if __name__ == "__main__": block to work with Python's multiprocessing module.
if __name__ == "__main__":
    run_finetuning()
