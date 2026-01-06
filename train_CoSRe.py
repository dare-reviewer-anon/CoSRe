# train_CoSRe.py
import os
import wandb
import torch
import logging
import argparse
import yaml
import torch.distributed as dist

from transformers import EarlyStoppingCallback, StopStringCriteria, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.generation import StoppingCriteriaList

from utils.run_config import create_run_name
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from utils.load_data import load_data, tokenize_dataset
from utils.load_model import load_model
from utils.evaluator import VisualizationEvaluator

logger = logging.getLogger(__name__)

# ==== W&B placeholders – replace with your real values if you want logging ====
WANDB_API_KEY = "<YOUR_WANDB_KEY_API>"
WANDB_ENTITY = "<YOUR_WANDB_ENTITY>"
PROJECT_NAME = "<YOUR_PROJECT_NAME>"


def get_rank_world():
    """Robust rank/world_size for both single-GPU and DDP."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def init_training_args(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    setting_type = "interleaved"
    with open(os.path.join(args.cfg_path, f"{setting_type}.yaml")) as f:
        training_cfg = yaml.safe_load(f)

    # override cfg if user passes CLI
    if args.train_bz:
        training_cfg["hyper"]["train_batch_size"] = args.train_bz
    if args.val_bz:
        training_cfg["hyper"]["val_batch_size"] = args.val_bz
    if args.grad_acc:
        training_cfg["hyper"]["grad_accumulation"] = args.grad_acc

    sup_hyper = training_cfg["hyper"]

    # Run name
    args.run_name = create_run_name(args, training_cfg)
    args.run_name = args.note + args.run_name

    training_args = WrappedSeq2SeqTrainingArguments(
        output_dir=os.path.join(args.output, args.run_name),
        remove_unused_columns=False,
        evaluation_strategy=training_cfg["eval"]["eval_strategy"],
        eval_steps=training_cfg["eval"]["eval_steps"]
        if training_cfg["eval"]["eval_strategy"] == "steps"
        else None,
        save_strategy=training_cfg["save"]["save_strategy"],
        save_steps=training_cfg["save"]["save_steps"]
        if training_cfg["save"]["save_strategy"] == "steps"
        else None,
        save_total_limit=40,
        seed=args.seed,
        # supervised tuning hyperparams
        learning_rate=sup_hyper["lr"],
        per_device_train_batch_size=sup_hyper["train_batch_size"],
        gradient_accumulation_steps=sup_hyper["grad_accumulation"],
        per_device_eval_batch_size=sup_hyper["val_batch_size"],
        num_train_epochs=sup_hyper["epochs"],
        logging_steps=training_cfg["logging"]["logging_step"],
        push_to_hub=False,
        predict_with_generate=training_cfg["model"]["predict_with_generate"],
        generation_max_new_tokens=training_cfg["model"]["generation_max_new_tokens"],
        generation_num_beams=training_cfg["model"]["generation_num_beams"],
    )

    # ===== memory-friendly settings =====
    training_args.bf16 = True
    training_args.fp16 = False
    training_args.gradient_checkpointing = True

    ds_config_path = os.path.join(args.cfg_path, "ds_zero3_4h100.json")
    if os.path.exists(ds_config_path):
        training_args.deepspeed = ds_config_path

    # ===== CoSRe flags stored in training_args for Trainer/Evaluator =====
    # CoSRe is inference-time KV-cache compression, so no extra training loss flags here.
    training_args.enable_CoSRe = args.enable_CoSRe
    training_args.cosre_block = args.cosre_block
    training_args.cosre_keep_h = args.cosre_keep_h
    training_args.cosre_keep_w = args.cosre_keep_w
    training_args.cosre_slots = args.cosre_slots
    training_args.cosre_base_delta = args.cosre_base_delta

    # ===== W&B =====
    rank, _ = get_rank_world()
    args.local_rank = rank

    if args.report_to == "wandb" and rank == 0:
        wandb.login(key=WANDB_API_KEY)
        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]

        wandb.init(
            project=os.getenv("WANDB_PROJECT", PROJECT_NAME),
            name=args.run_name,
            entity=os.getenv("WANDB_ENTITY", WANDB_ENTITY),
            **init_args,
        )
        wandb.config.update(training_args, allow_val_change=True)
    else:
        training_args.report_to = []

    # ===== checkpoint handling =====
    # if user points to a finetuned checkpoint, we resume from it
    if os.path.exists(training_args.output_dir) and args.model_ckpt is None:
        args.model_ckpt = training_args.output_dir

    if args.model_ckpt is not None:
        training_args.load_weights_from = get_last_checkpoint(args.model_ckpt)
    else:
        training_args.load_weights_from = None

    return training_args


def configure_model_for_train_eval(model, do_train: bool):
    """
    Training: disable KV cache (saves memory); Eval/Predict: enable KV cache.
    """
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = (not do_train)


def maybe_enable_model_checkpointing(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="anole")
    parser.add_argument("--data", type=str, nargs="+", required=True)
    parser.add_argument("--data_dir", type=str, default="data_samples")
    parser.add_argument("--decoder_type", type=str, default="anole")
    parser.add_argument("--note", type=str, default="debug-")
    parser.add_argument("--image_seq_length", type=int, default=1024)
    parser.add_argument("--no_perceptual_loss", action="store_true")

    # model ckpt
    parser.add_argument("--model_ckpt", type=str, default=None, help="Path to a checkpoint dir.")
    parser.add_argument("--load_last_checkpoint", action="store_true")

    # train/eval/predict
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--cfg_path", type=str, default="cfg")
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--input_format", type=str, default="anole")

    # output/logging
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--cache_dir", type=str, default=None)

    # seed/ddp
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int)

    # debug
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--train_bz", type=int, default=None)
    parser.add_argument("--val_bz", type=int, default=None)
    parser.add_argument("--grad_acc", type=int, default=None)

    # ====== CoSRe-specific CLI flags (inference-time) ======
    parser.add_argument(
        "--enable_CoSRe",
        action="store_true",
        help="Enable CoSRe decoding / KV-cache compression during eval/predict.",
    )
    parser.add_argument("--cosre_block", type=int, default=8, help="Block size b for blockwise cosine compression.")
    parser.add_argument("--cosre_keep_h", type=int, default=4, help="Keep low-freq spatial height per block.")
    parser.add_argument("--cosre_keep_w", type=int, default=4, help="Keep low-freq spatial width per block.")
    parser.add_argument("--cosre_slots", type=int, default=32, help="Number of shared semantic slots K.")
    parser.add_argument("--cosre_base_delta", type=float, default=0.5, help="Base quant step for Delta(u,v).")

    args = parser.parse_args()

    if args.model in ["anole"]:
        args.decoder_type = args.model
        assert args.input_format == "anole"

    if args.decoder_type in ["anole"]:
        args.note = args.note + f"image_seq_len-{args.image_seq_length}-"

    training_args = init_training_args(args)

    # ===== Load data =====
    print(f"Preparing the {args.data} dataset... ")
    data = load_data(dataset=args.data, data_dir=args.data_dir)

    if len(data) == 2:
        train_split, eval_split, test_split = data["train"], None, data["test"]
    else:
        try:
            train_split, eval_split, test_split = data["train"], data["dev"], data["test"]
        except Exception:
            train_split, eval_split, test_split = data["train"], data["validation"], data["test"]

    if args.toy:
        print("Only using toy examples for debugging...")
        max_train_toy, max_eval_toy, max_test_toy = 100, 10, 10
        train_split = train_split.select(list(range(min(max_train_toy, len(train_split)))))
        if eval_split is not None:
            eval_split = eval_split.select(list(range(min(max_eval_toy, len(eval_split)))))
        if test_split is not None:
            test_split = test_split.select(list(range(min(max_test_toy, len(test_split)))))

    # ===== Load model =====
    model_processor = load_model(args)
    model, processor = model_processor["model"], model_processor["processor"]

    maybe_enable_model_checkpointing(model)

    # Training should not use KV cache; eval/predict should.
    # We’ll switch right before calling evaluate/predict.
    configure_model_for_train_eval(model, do_train=args.do_train)

    # ===== Tokenize =====
    rank, world_size = get_rank_world()

    if eval_split is not None:
        # make eval/test divisible by global batch size for DDP
        gbs_eval = training_args.per_device_eval_batch_size * world_size
        eval_data_num = (len(eval_split) // gbs_eval) * gbs_eval
        test_data_num = (len(test_split) // gbs_eval) * gbs_eval
        eval_split = eval_split.select(list(range(eval_data_num)))
        test_split = test_split.select(list(range(test_data_num)))
        print(f"Eval Num: {eval_data_num}")
    else:
        print("No eval split detected; skipping eval truncation.")

    tokenized_data, max_source_length, max_target_length = tokenize_dataset(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        model=model,
        processor=processor,
        input_format=args.input_format,
        interleave=True,
        data_name="-".join(args.data),
    )

    training_args.generation_max_new_tokens = max_target_length + 100
    print(f"generation_max_new_tokens: {training_args.generation_max_new_tokens}")

    # ===== Data collator & trainer =====
    from utils.data_collator import customize_data_collator
    from utils.trainer.customize_trainer import CustomizeSeq2SeqTrainer

    training_args.label_smoothing_factor = 0.1

    kwargs = {}
    if args.model in ["anole"]:
        kwargs["multimodal_generation_mode"] = "interleaved-text-image"
        kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)]
        )
        training_args.customize_gen_stopping_criteria = kwargs["stopping_criteria"]

    trainer = CustomizeSeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=VisualizationEvaluator(args=args),
        tokenizer=processor,
        data_collator=customize_data_collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["eval"] if "eval" in tokenized_data else tokenized_data["test"],
        eval_examples=eval_split if "eval" in tokenized_data else test_split,
        wandb_run_dir=wandb.run.dir if (args.report_to == "wandb" and rank == 0) else None,
        image_loss_func=not args.no_perceptual_loss,
    )

    print("CoSRe Trainer built successfully.")

    # ===== OPTIONAL: Attach CoSRe decoding hook =====
    # This does NOT affect training. It only affects evaluate()/predict() generation.
    if training_args.enable_CoSRe:
        try:
            # implement this function in your repo OR ignore this block and handle inside Trainer
            from utils.cosre_hook import attach_cosre_for_generation
            attach_cosre_for_generation(trainer=trainer, processor=processor)
            logger.info("CoSRe decoding hook attached (eval/predict only).")
        except Exception as e:
            logger.warning(
                f"enable_CoSRe=True but utils.cosre_hook.attach_cosre_for_generation() not found/failed: {e}. "
                "Make sure CustomizeSeq2SeqTrainer reads training_args.enable_CoSRe and switches decoding."
            )

    checkpoint = training_args.load_weights_from

    # ===== Train =====
    if args.do_train:
        # KV cache OFF already
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_data["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ===== Eval / Predict =====
    if args.do_eval:
        # CoSRe needs cache ON during generation/eval
        configure_model_for_train_eval(trainer.model, do_train=False)

        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", **kwargs)

        if metrics:
            if "eval" in tokenized_data:
                metrics["eval_samples"] = len(tokenized_data["eval"])
            else:
                metrics["eval_samples"] = len(tokenized_data["test"])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if args.do_predict:
            logger.info("*** Predict ***")
            predict_results = trainer.predict(
                test_dataset=tokenized_data["test"],
                test_examples=tokenized_data["test"].dataset,
                metric_key_prefix="predict",
                **kwargs,
            )
            pmetrics = predict_results.metrics
            if pmetrics:
                pmetrics["predict_samples"] = len(tokenized_data["test"])
                trainer.log_metrics("predict", pmetrics)
                trainer.save_metrics("predict", pmetrics)


if __name__ == "__main__":
    main()
