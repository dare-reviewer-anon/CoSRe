import os
import wandb
import torch
import logging
import argparse
import yaml
import torch.distributed as dist

from transformers import (
    EarlyStoppingCallback,
    StopStringCriteria,
    set_seed,
)
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


def init_training_args(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.local_rank is not None:
        torch.cuda.set_device(args.local_rank)

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    setting_type = "interleaved"
    with open(os.path.join(args.cfg_path, setting_type + ".yaml")) as f:
        training_cfg = yaml.safe_load(f)

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
        learning_rate=sup_hyper["lr"] if sup_hyper else 0,
        per_device_train_batch_size=sup_hyper["train_batch_size"] if sup_hyper else 0,
        gradient_accumulation_steps=sup_hyper["grad_accumulation"] if sup_hyper else 0,
        per_device_eval_batch_size=sup_hyper["val_batch_size"]
        if sup_hyper
        else training_cfg["hyper"]["val_batch_size"],
        num_train_epochs=sup_hyper["epochs"] if sup_hyper else 0,
        logging_steps=training_cfg["logging"]["logging_step"],
        push_to_hub=False,
        predict_with_generate=training_cfg["model"]["predict_with_generate"],
        generation_max_new_tokens=training_cfg["model"]["generation_max_new_tokens"],
        generation_num_beams=training_cfg["model"]["generation_num_beams"],
    )

    # ===== Your memory-friendly settings (kept as in your script) =====
    training_args.bf16 = True
    training_args.fp16 = False
    training_args.gradient_checkpointing = True

    ds_config_path = os.path.join(args.cfg_path, "ds_zero3_4A100.json")
    if os.path.exists(ds_config_path):
        training_args.deepspeed = ds_config_path

    # ===== DARE-specific metadata stored inside training_args (for Trainer/use later) =====
    training_args.enable_dare = args.enable_dare
    training_args.rho_text_target = args.rho_text_target
    training_args.rho_vis_target = args.rho_vis_target
    training_args.dare_tau = args.dare_tau
    training_args.dare_lambda_ratio = args.dare_lambda_ratio
    training_args.dare_lambda_soft = args.dare_lambda_soft
    training_args.dare_lambda_hard = args.dare_lambda_hard
    training_args.dare_prefix_kappa = args.dare_prefix_kappa

    # ===== DDP / rank detection =====
    try:
        rank = dist.get_rank()
    except Exception:
        rank = 0
    args.local_rank = rank

    # ===== W&B setup (same logic as your script, but slightly cleaned) =====
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

    # ===== checkpoint handling: we reuse this to start from MVoT-finetuned model =====
    if os.path.exists(training_args.output_dir) and args.model_ckpt is None:
        # if output dir exists and no explicit model_ckpt given, treat it as resume dir
        args.model_ckpt = training_args.output_dir

    if args.model_ckpt is not None:
        training_args.load_weights_from = get_last_checkpoint(args.model_ckpt)
    else:
        training_args.load_weights_from = None

    return training_args


def maybe_attach_dare(model, training_args):
    """
    Attach DARE router / configs to the model.

    NOTE: this assumes you have implemented DARE modules somewhere like:
      model_utils/dare/router.py, model_utils/dare/wrappers.py, etc.

    The key design: we DO NOT change the training loop or HF Trainer;
    we just (1) add small router modules and (2) let the model's forward()
    consume training_args.enable_dare & co.
    """
    if not getattr(training_args, "enable_dare", False):
        return model  # vanilla MVoT behaviour

    try:
        from model_utils.dare import DAREConfig, DAREController, attach_dare_to_anole
    except ImportError:
        logger.warning(
            "DARE is enabled but model_utils.dare is not found. "
            "Please create model_utils/dare/__init__.py with DAREConfig, "
            "DAREController, and attach_dare_to_anole()."
        )
        return model

    cfg = DAREConfig(
        hidden_size=getattr(model.config, "hidden_size", None),
        num_layers=getattr(model.config, "num_hidden_layers", None),
        num_heads=getattr(model.config, "num_attention_heads", None),
        rho_text_target=training_args.rho_text_target,
        rho_vis_target=training_args.rho_vis_target,
        tau=training_args.dare_tau,
        lambda_ratio=training_args.dare_lambda_ratio,
        lambda_soft=training_args.dare_lambda_soft,
        lambda_hard=training_args.dare_lambda_hard,
        prefix_kappa=training_args.dare_prefix_kappa,
    )
    controller = DAREController(cfg)

    # This function is where you actually wrap the Anole blocks
    # so that they call into the router and apply masks / KV pruning.
    model = attach_dare_to_anole(model, controller)

    logger.info(
        f"DARE attached: rho_t={cfg.rho_text_target}, "
        f"rho_v={cfg.rho_vis_target}, tau={cfg.tau}"
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="anole")
    parser.add_argument("--data", type=str, nargs="+")
    parser.add_argument("--data_dir", type=str, default="data_samples")
    parser.add_argument("--decoder_type", type=str, default="anole")
    parser.add_argument("--note", type=str, default="debug")
    parser.add_argument("--image_seq_length", type=int, default=1024)
    parser.add_argument("--no_perceptual_loss", action="store_true")

    # model ckpt: MUST point to the MVoT-finetuned run for our method
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint dir (use MVoT-finetuned dir for DARE).",
    )
    parser.add_argument("--load_last_checkpoint", action="store_true")

    # training / evaluation flags
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--cfg_path", type=str, default="cfg")
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--input_format", type=str, default="anole")

    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--toy", action="store_true")

    parser.add_argument("--train_bz", type=int, default=None)
    parser.add_argument("--val_bz", type=int, default=None)
    parser.add_argument("--grad_acc", type=int, default=None)

    # ====== DARE-specific CLI flags ======
    parser.add_argument(
        "--enable_dare",
        action="store_true",
        help="Enable DARE routing / pruning on top of MVoT.",
    )
    parser.add_argument(
        "--rho_text_target",
        type=float,
        default=0.7,
        help="Global target retention ratio for text tokens.",
    )
    parser.add_argument(
        "--rho_vis_target",
        type=float,
        default=0.4,
        help="Global target retention ratio for visual tokens.",
    )
    parser.add_argument(
        "--dare_tau",
        type=float,
        default=0.5,
        help="Gumbel-Softmax temperature.",
    )
    parser.add_argument(
        "--dare_lambda_ratio",
        type=float,
        default=1.0,
        help="Weight for ratio-matching loss L_ratio^t.",
    )
    parser.add_argument(
        "--dare_lambda_soft",
        type=float,
        default=1.0,
        help="Weight for visual soft sparsity loss L_soft^v.",
    )
    parser.add_argument(
        "--dare_lambda_hard",
        type=float,
        default=1.0,
        help="Weight for visual hard penalty L_hard^v.",
    )
    parser.add_argument(
        "--dare_prefix_kappa",
        type=int,
        default=16,
        help="Number of prefix tokens per hop always retained in KV cache.",
    )

    args = parser.parse_args()

    if args.model in ["anole"]:
        args.decoder_type = args.model
        assert args.input_format == "anole"

    if args.decoder_type in ["anole"]:
        args.note = args.note + f"image_seq_len-{str(args.image_seq_length)}-"

    training_args = init_training_args(args)

    print(f"Preparing the {args.data} dataset... ")
    data = load_data(dataset=args.data, data_dir=args.data_dir)

    if len(data) == 2:
        train_split, eval_split, test_split = data["train"], None, data["test"]
    else:
        try:
            train_split, eval_split, test_split = (
                data["train"],
                data["dev"],
                data["test"],
            )
        except Exception:
            train_split, eval_split, test_split = (
                data["train"],
                data["validation"],
                data["test"],
            )

    if args.toy:
        print("Only using toy examples for debugging...")
        max_train_toy = 100
        max_eval_toy = 10
        max_test_toy = 10

        n_train = min(max_train_toy, len(train_split))
        train_split = train_split.select(list(range(n_train)))

        if eval_split is not None:
            n_eval = min(max_eval_toy, len(eval_split))
            eval_split = eval_split.select(list(range(n_eval)))

        if test_split is not None:
            n_test = min(max_test_toy, len(test_split))
            test_split = test_split.select(list(range(n_test)))

    # ===== Load MVoT / Anole model (possibly from MVoT-finetuned ckpt) =====
    model_processor = load_model(args)
    model, processor = model_processor["model"], model_processor["processor"]

    # Trainer-side gradient checkpointing already enabled via training_args,
    # but we also enable model-side checkpointing where supported.
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing = True

    # Disable KV cache during training for memory
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # ===== Attach DARE routing and KV-cache pruning (no change if enable_dare=False) =====
    model = maybe_attach_dare(model, training_args)

    # ===== Dataset tokenization =====
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if eval_split is not None:

        eval_data_num = (
            len(eval_split)
            // (training_args.per_device_eval_batch_size * world_size)
        ) * (training_args.per_device_eval_batch_size * world_size)
        eval_split = eval_split.select(list(range(eval_data_num)))

        test_data_num = (
            len(test_split)
            // (training_args.per_device_eval_batch_size * world_size)
        ) * (training_args.per_device_eval_batch_size * world_size)
        test_split = test_split.select(list(range(test_data_num)))

        print(f"Eval Num: {eval_data_num}")

    else:
        eval_data_num = 0
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

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience
    )

    from utils.data_collator import customize_data_collator
    data_collator = customize_data_collator

    from utils.trainer.customize_trainer import CustomizeSeq2SeqTrainer
    trainer_type = CustomizeSeq2SeqTrainer

    training_args.label_smoothing_factor = 0.1

    if args.model in ["anole"]:
        kwargs = dict()
        kwargs["multimodal_generation_mode"] = "interleaved-text-image"
        kwargs["stopping_criteria"] = StoppingCriteriaList(
            [
                StopStringCriteria(
                    stop_strings=["<reserved08706>", "</s>"],
                    tokenizer=processor.tokenizer,
                )
            ]
        )
        training_args.customize_gen_stopping_criteria = StoppingCriteriaList(
            [
                StopStringCriteria(
                    stop_strings=["<reserved08706>", "</s>"],
                    tokenizer=processor.tokenizer,
                )
            ]
        )

    trainer = trainer_type(
        args=training_args,
        model=model,
        evaluator=VisualizationEvaluator(args=args),
        tokenizer=processor,
        data_collator=data_collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["eval"]
        if "eval" in tokenized_data.keys()
        else tokenized_data["test"],
        eval_examples=eval_split
        if "eval" in tokenized_data.keys()
        else test_split,
        wandb_run_dir=wandb.run.dir
        if (args.report_to == "wandb" and args.local_rank == 0)
        else None,
        image_loss_func=not args.no_perceptual_loss,
    )

    print("DARE Trainer built successfully.")

    checkpoint = training_args.load_weights_from

    if args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = len(tokenized_data["train"])
        metrics["train_samples"] = min(
            max_train_samples, len(tokenized_data["train"])
        )

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", **kwargs)

        if metrics is not None and len(metrics) > 0:
            if "eval" in tokenized_data:
                max_eval_samples = len(tokenized_data["eval"])
                metrics["eval_samples"] = min(
                    max_eval_samples, len(tokenized_data["eval"])
                )
            else:
                max_eval_samples = len(tokenized_data["test"])
                metrics["eval_samples"] = min(
                    max_eval_samples, len(tokenized_data["test"])
                )

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
            metrics = predict_results.metrics
            max_predict_samples = len(tokenized_data["test"])
            metrics["predict_samples"] = min(
                max_predict_samples, len(tokenized_data["test"])
            )

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)
