import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
from config_module import (
    model_config,
    train_config,
    lora_config,
    data_config,
)


def main():
    # 创建 parser 并指定数据类
    parser = HfArgumentParser(
        (model_config, train_config, lora_config, data_config)
    )

    # 从命令行解析参数到数据类
    (
        model_args,
        train_args,
        lora_args,
        data_args,
    ) = parser.parse_args_into_dataclasses()

    args = argparse.Namespace(
        **vars(model_args), **vars(train_args), **vars(lora_args), **vars(data_args)
    )

    if args.xformers:        # 开启xformers设置
        from llama_xformers_attn_monkey_patch import replace_llama_attn_with_xformers_attn
        replace_llama_attn_with_xformers_attn()

    device_map = "auto"
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    compute_dtype = (
        torch.float16
        if args.fp16
        else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 模型导入
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        cache_dir=args.cache_dir,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        quantization_config=bnb_config,
        # use_auth_token=args.use_auth_token,
    )

    # lora_config设置导入
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,  # 将其设置为"none"，以仅训练权重参数而不是偏差
        task_type="CAUSAL_LM",
    )

    #  获得lora合并后的当前模型
    # model_lora = get_peft_model(model=model, peft_config=lora_config)

    #  打印当前可训练参数情况
    # model_lora.print_trainable_parameters()

    # if args.gradient_checkpointing:
    #     model_lora.enable_input_require_grads()

    # if not ddp and torch.cuda.device_count() > 1:
    #     # 是否开启并行操作
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    # 加载tokenizer方法
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        cache_dir=args.cache_dir,
        use_fast=False,
        padding_side="right",
        tokenizer_type='llama',
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token  # llama缺失的pad补充为eos值

    dataset = load_dataset("json", data_files=args.data_path)  # 以dataset["train"]形式导入数据集

    if args.eval_dataset_size > 0:
        train_val = dataset["train"].train_test_split(
            test_size=args.eval_dataset_size, shuffle=True, seed=0
        )
        train_data = (
            train_val["train"].shuffle()
        )
        val_data = (
            train_val["test"].shuffle()
        )
    else:
        train_data = dataset["train"].shuffle()
        val_data = None

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")

    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        peft_config=peft_config,  # peft-lora参数
        train_dataset=train_data,
        eval_dataset=val_data,
        args=train_args,
        max_seq_length=args.model_max_length,
        packing=False,
        padding=True,
        dataset_text_field="text",
    )
    model.config.use_cache = args.use_cache

    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
