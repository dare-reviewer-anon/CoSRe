# train_CoSRe.py
import os
import logging
import argparse
import yaml
import torch
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


# =========================
# DDP helpers
# =========================
def get_rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def is_rank0() -> bool:
    r, _ = get_rank_world()
    return r == 0


# =========================
# Model toggles
# =========================
def set_model_cache(model, use_cache: bool):
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = bool(use_cache)


def do_enable_model_checkpointing(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing = True


# =========================
# Training args init
# =========================
def init_training_args(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    # Load config
    setting_type = "interleaved"
    with open(os.path.join(args.cfg_path, f"{setting_type}.yaml"), "r") as f:
        training_cfg = yaml.safe_load(f)

    # Override cfg if CLI specifies
    if args.train_bz is not None:
        training_cfg["hyper"]["train_batch_size"] = args.train_bz
    if args.val_bz is not None:
        training_cfg["hyper"]["val_batch_size"] = args.val_bz
    if args.grad_acc is not None:
        training_cfg["hyper"]["grad_accumulation"] = args.grad_acc

    sup_hyper = training_cfg["hyper"]

    # Run name
    args.run_name = args.note + create_run_name(args, training_cfg)

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
        learning_rate=float(sup_hyper["lr"]),
        per_device_train_batch_size=int(sup_hyper["train_batch_size"]),
        gradient_accumulation_steps=int(sup_hyper["grad_accumulation"]),
        per_device_eval_batch_size=int(sup_hyper["val_batch_size"]),
        num_train_epochs=float(sup_hyper["epochs"]),
        logging_steps=int(training_cfg["logging"]["logging_step"]),
        push_to_hub=False,
        predict_with_generate=bool(training_cfg["model"]["predict_with_generate"]),
        generation_max_new_tokens=int(training_cfg["model"]["generation_max_new_tokens"]),
        generation_num_beams=int(training_cfg["model"]["generation_num_beams"]),
    )

    # Memory-friendly defaults
    training_args.bf16 = True
    training_args.fp16 = False
    training_args.gradient_checkpointing = True

    # DeepSpeed config: prefer user arg, else try common names
    ds_cfg = args.deepspeed_config
    if ds_cfg is None:
        for cand in ["ds_zero3_4a100.json", "ds_zero3_4h100.json", "ds_zero3_4A100.json", "ds_zero3_4H100.json"]:
            p = os.path.join(args.cfg_path, cand)
            if os.path.exists(p):
                ds_cfg = p
                break
    if ds_cfg is not None and os.path.exists(ds_cfg):
        training_args.deepspeed = ds_cfg

    # ===== CoSRe flags saved into training_args (for CustomizeSeq2SeqTrainer) =====
    training_args.enable_CoSRe = bool(args.enable_CoSRe)
    training_args.cosre_block = int(args.cosre_block)
    training_args.cosre_keep_h = int(args.cosre_keep_h)
    training_args.cosre_keep_w = int(args.cosre_keep_w)
    training_args.cosre_slots = int(args.cosre_slots)
    training_args.cosre_base_delta = float(args.cosre_base_delta)

    # Report-to: disable if requested
    if args.report_to in [None, "", "none", "None"]:
        training_args.report_to = []
    else:
        training_args.report_to = [args.report_to]

    # Checkpoint selection:
    # - model_ckpt: baseline checkpoint dir
    # - load_last_checkpoint: resume from last checkpoint within model_ckpt
    if args.model_ckpt is not None and args.load_last_checkpoint:
        training_args.load_weights_from = get_last_checkpoint(args.model_ckpt)
    else:
        training_args.load_weights_from = None

    return training_args


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # core
    parser.add_argument("--model", type=str, default="anole")
    parser.add_argument("--data", type=str, nargs="+", required=True)
    parser.add_argument("--data_dir", type=str, default="data_samples")
    parser.add_argument("--decoder_type", type=str, default="anole")
    parser.add_argument("--input_format", type=str, default="anole")
    parser.add_argument("--image_seq_length", type=int, default=1024)
    parser.add_argument("--note", type=str, default="cosre-")

    # stage flags
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # config/output
    parser.add_argument("--cfg_path", type=str, default="cfg")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--cache_dir", type=str, default=None)

    # checkpoint
    parser.add_argument("--model_ckpt", type=str, default=None, help="Baseline checkpoint dir to load.")
    parser.add_argument("--load_last_checkpoint", action="store_true", help="Resume from last checkpoint in model_ckpt.")

    # runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=None)

    # memory / deepspeed
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to deepspeed json, optional.")

    # debug / batch
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--train_bz", type=int, default=None)
    parser.add_argument("--val_bz", type=int, default=None)
    parser.add_argument("--grad_acc", type=int, default=None)

    # perceptual loss
    parser.add_argument("--no_perceptual_loss", action="store_true")

    # ===== CoSRe inference-time params =====
    parser.add_argument("--enable_CoSRe", action="store_true")
    parser.add_argument("--cosre_block", type=int, default=8)
    parser.add_argument("--cosre_keep_h", type=int, default=4)
    parser.add_argument("--cosre_keep_w", type=int, default=4)
    parser.add_argument("--cosre_slots", type=int, default=32)
    parser.add_argument("--cosre_base_delta", type=float, default=0.5)

    args = parser.parse_args()

    # model constraints
    if args.model == "anole":
        args.decoder_type = "anole"
        assert args.input_format == "anole"

    args.note = args.note + f"image_seq_len-{args.image_seq_length}-"

    # training args
    training_args = init_training_args(args)

    # ===== Load data =====
    if is_rank0():
        print(f"Preparing dataset: {args.data} from {args.data_dir}")
    data = load_data(dataset=args.data, data_dir=args.data_dir)

    if len(data) == 2:
        train_split, eval_split, test_split = data["train"], None, data["test"]
    else:
        try:
            train_split, eval_split, test_split = data["train"], data["dev"], data["test"]
        except Exception:
            train_split, eval_split, test_split = data["train"], data["validation"], data["test"]

    if args.toy:
        if is_rank0():
            print("Using toy subset for debugging...")
        train_split = train_split.select(list(range(min(100, len(train_split)))))
        if eval_split is not None:
            eval_split = eval_split.select(list(range(min(10, len(eval_split)))))
        test_split = test_split.select(list(range(min(10, len(test_split)))))

    # ===== Load model =====
    model_processor = load_model(args)
    model, processor = model_processor["model"], model_processor["processor"]

    do_enable_model_checkpointing(model)

    # Training: cache OFF; Eval/Predict: we will switch to ON before generating
    set_model_cache(model, use_cache=not args.do_train)

    # ===== Tokenize =====
    rank, world = get_rank_world()
    if eval_split is not None:
        # Make eval/test divisible by global eval batch size for DDP
        gbs = training_args.per_device_eval_batch_size * world
        eval_n = (len(eval_split) // gbs) * gbs
        test_n = (len(test_split) // gbs) * gbs
        eval_split = eval_split.select(list(range(eval_n)))
        test_split = test_split.select(list(range(test_n)))
        if is_rank0():
            print(f"Eval Num (DDP-aligned): {eval_n}")

    tokenized_data, _, max_target_length = tokenize_dataset(
        train_split=train_split,
        eval_split=eval_split,
        test_split=test_split,
        model=model,
        processor=processor,
        input_format=args.input_format,
        interleave=True,
        data_name="-".join(args.data),
    )

    training_args.generation_max_new_tokens = int(max_target_length) + 100
    if is_rank0():
        print(f"generation_max_new_tokens: {training_args.generation_max_new_tokens}")

    # ===== CoSRe stopping criteria for anole =====
    kwargs = {}
    if args.model == "anole":
        kwargs["multimodal_generation_mode"] = "interleaved-text-image"
        stop = StoppingCriteriaList(
            [StopStringCriteria(stop_strings=["<reserved08706>", "</s>"], tokenizer=processor.tokenizer)]
        )
        kwargs["stopping_criteria"] = stop
        training_args.customize_gen_stopping_criteria = stop

    # ===== Trainer =====
    from utils.data_collator import customize_data_collator
    from utils.trainer.customize_trainer import CustomizeSeq2SeqTrainer

    training_args.label_smoothing_factor = 0.1

    trainer = CustomizeSeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=VisualizationEvaluator(args=args),
        tokenizer=processor,
        data_collator=customize_data_collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["eval"] if "eval" in tokenized_data else tokenized_data["test"],
        eval_examples=eval_split if "eval" in tokenized_data else test_split,
        wandb_run_dir=None,  # keep None unless your Trainer explicitly needs it
        image_loss_func=not args.no_perceptual_loss,
    )

    if is_rank0():
        print("CoSRe Trainer built successfully.")
        if training_args.enable_CoSRe:
            print(
                f"CoSRe enabled for eval/predict: block={training_args.cosre_block}, "
                f"keep=({training_args.cosre_keep_h},{training_args.cosre_keep_w}), "
                f"slots={training_args.cosre_slots}, delta={training_args.cosre_base_delta}"
            )

    # resume checkpoint (training)
    checkpoint = training_args.load_weights_from

    # ===== Train =====
    if args.do_train:
        # ensure cache OFF
        set_model_cache(trainer.model, use_cache=False)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_data["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ===== Eval =====
    if args.do_eval:
        # CoSRe benefits require cache ON during generation
        set_model_cache(trainer.model, use_cache=True)

        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", **kwargs)

        if metrics and is_rank0():
            n = len(tokenized_data["eval"]) if "eval" in tokenized_data else len(tokenized_data["test"])
            metrics["eval_samples"] = n
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    # ===== Predict =====
    if args.do_predict:
        set_model_cache(trainer.model, use_cache=True)

        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            test_dataset=tokenized_data["test"],
            test_examples=tokenized_data["test"].dataset,
            metric_key_prefix="predict",
            **kwargs,
        )
        pmetrics = predict_results.metrics
        if pmetrics and is_rank0():
            pmetrics["predict_samples"] = len(tokenized_data["test"])
            trainer.log_metrics("predict", pmetrics)
            trainer.save_metrics("predict", pmetrics)


if __name__ == "__main__":
    main()
