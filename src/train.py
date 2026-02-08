import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from .model import (
    CoTProtocol,
    DirectProtocol,
    ECSRProtocol,
    ModelWrapper,
    ProtocolResult,
    SRResetProtocol,
    load_model_and_tokenizer,
    numeric_equal,
)
from .preprocess import ReasoningDataset, load_dataset_splits

LOGGED_METRICS = [
    "accuracy",
    "mean_abs_error",
    "parse_success_rate",
    "state_valid_rate",
    "repair_rate",
    "avg_model_calls",
    "avg_output_tokens",
    "avg_repairs_per_problem",
    "example_correct",
    "example_parse_success",
    "example_abs_error",
    "example_model_calls",
    "example_output_tokens",
    "example_repairs_used",
    "example_state_valid_rate",
    "example_complexity",
    "train_loss",
    "train_grad_norm",
    "epoch",
]


def cfg_get(cfg_obj: Any, key: str, default: Any = None) -> Any:
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, (DictConfig, dict)):
        return cfg_obj.get(key, default)
    if hasattr(cfg_obj, "get"):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def resolve_run_cfg(cfg: DictConfig) -> DictConfig:
    if "run" in cfg and isinstance(cfg.run, (DictConfig, dict)):
        return cfg.run
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_run_id(cfg: DictConfig) -> str:
    if "run" in cfg and isinstance(cfg.run, (dict, DictConfig)):
        run_id = cfg.run.get("run_id")
        if run_id:
            return str(run_id)
    run_id = cfg.get("run_id")
    if run_id:
        return str(run_id)
    raise ValueError("run_id is missing; ensure run=<run_id> is provided via Hydra.")


def apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    run_cfg = resolve_run_cfg(cfg)
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.trial_max_batches = 2
        cfg.trial_max_examples = 2
        if "optuna" in run_cfg and run_cfg.optuna is not None:
            run_cfg.optuna.n_trials = 0
        elif "optuna" in cfg:
            cfg.optuna.n_trials = 0
        if "training" in run_cfg and run_cfg.training is not None:
            run_cfg.training.epochs = max(1, int(cfg_get(run_cfg.training, "epochs", 1)))
        elif "training" in cfg:
            cfg.training.epochs = max(1, int(cfg_get(cfg.training, "epochs", 1)))
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")
    return cfg


def build_protocol(run_cfg: DictConfig, model_wrapper: ModelWrapper):
    method = str(cfg_get(run_cfg, "method", "direct")).lower()
    protocol_cfg = cfg_get(run_cfg, "protocol", None)
    if protocol_cfg is None:
        raise ValueError("protocol configuration is missing")
    decoding_cfg = cfg_get(protocol_cfg, "decoding", {})
    if "ecsr" in method:
        return ECSRProtocol(model_wrapper, protocol_cfg, decoding_cfg)
    if "sr" in method or "reset" in method:
        return SRResetProtocol(model_wrapper, protocol_cfg, decoding_cfg)
    if "cot" in method:
        return CoTProtocol(model_wrapper, protocol_cfg, decoding_cfg)
    return DirectProtocol(model_wrapper, protocol_cfg, decoding_cfg)


def build_dataloader(dataset: ReasoningDataset, batch_size: int) -> DataLoader:
    def collate_fn(batch: List[Any]) -> Dict[str, Any]:
        return {
            "questions": [ex.question for ex in batch],
            "gold_values": [ex.gold_value for ex in batch],
            "example_ids": [ex.example_id for ex in batch],
            "complexities": [ex.complexity for ex in batch],
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)


class Seq2SeqDataset(Dataset):
    def __init__(self, examples: ReasoningDataset):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        ex = self.examples[idx]
        return {"question": ex.question, "answer": ex.gold_text}


def build_seq2seq_dataloader(
    dataset: ReasoningDataset, tokenizer, batch_size: int, max_length: int
) -> DataLoader:
    seq_dataset = Seq2SeqDataset(dataset)

    def collate_fn(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]
        inputs = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        targets = tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = targets["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

    return DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)


def shift_tokens_right(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: Optional[int]) -> torch.Tensor:
    if decoder_start_token_id is None:
        raise ValueError("decoder_start_token_id must be set for shift_right")
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def assert_gradients(model: torch.nn.Module) -> None:
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "No gradients found before optimizer.step()"
    non_zero = any((g is not None and torch.any(g != 0)) for g in grads)
    assert non_zero, "All gradients are zero before optimizer.step()"


def compute_accuracy_by_bin(bin_totals: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    results = {}
    for bin_name, stats in bin_totals.items():
        total = max(stats["total"], 1)
        results[f"accuracy_{bin_name}"] = stats["correct"] / total
    return results


def evaluate_dataset(
    protocol,
    dataloader: DataLoader,
    dataset_cfg: DictConfig,
    cfg: DictConfig,
    tokenizer,
    log_to_wandb: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    total_examples = 0
    correct = 0
    parse_success = 0
    abs_error_sum = 0.0
    model_calls_sum = 0
    output_tokens_sum = 0
    repairs_sum = 0
    repairs_triggered = 0
    state_valid_rounds = 0
    total_rounds = 0
    bin_totals = {
        "bin_1_2": {"correct": 0, "total": 0},
        "bin_3_4": {"correct": 0, "total": 0},
        "bin_5_plus": {"correct": 0, "total": 0},
    }

    global_step = 0
    preproc = cfg_get(dataset_cfg, "preprocessing", {})
    max_length = int(cfg_get(preproc, "max_length", 512))

    if max_batches is not None:
        max_batches = int(max_batches)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        questions = batch["questions"]
        gold_values = batch["gold_values"]
        complexities = batch["complexities"]

        if batch_idx == 0:
            tokenized = tokenizer(
                questions,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            assert tokenized["input_ids"].shape[0] == len(questions)
            assert len(questions) == len(gold_values)

        for question, gold, complexity in zip(questions, gold_values, complexities):
            result: ProtocolResult = protocol.run(question)
            total_examples += 1
            model_calls_sum += result.total_calls
            output_tokens_sum += result.total_new_tokens
            repairs_sum += result.repairs_used
            if result.repairs_used > 0:
                repairs_triggered += 1
            state_valid_rounds += result.state_valid_rounds
            total_rounds += result.total_rounds

            pred_value = result.prediction_value
            is_parse_success = pred_value is not None
            if is_parse_success:
                parse_success += 1
                abs_error = abs(pred_value - gold)
                abs_error_sum += abs_error
                is_correct = numeric_equal(pred_value, gold)
            else:
                abs_error = float("nan")
                is_correct = False

            if is_correct:
                correct += 1

            if complexity <= 2:
                bin_name = "bin_1_2"
            elif complexity <= 4:
                bin_name = "bin_3_4"
            else:
                bin_name = "bin_5_plus"
            bin_totals[bin_name]["total"] += 1
            bin_totals[bin_name]["correct"] += int(is_correct)

            running_accuracy = correct / total_examples
            running_parse_success = parse_success / total_examples
            running_mae = abs_error_sum / parse_success if parse_success > 0 else float("nan")
            running_avg_calls = model_calls_sum / total_examples
            running_avg_tokens = output_tokens_sum / total_examples
            running_repair_rate = repairs_triggered / total_examples
            running_state_valid_rate = state_valid_rounds / total_rounds if total_rounds > 0 else 0.0
            running_avg_repairs = repairs_sum / total_examples

            if log_to_wandb and cfg.wandb.mode != "disabled":
                wandb.log(
                    {
                        "accuracy": running_accuracy,
                        "mean_abs_error": running_mae,
                        "parse_success_rate": running_parse_success,
                        "avg_model_calls": running_avg_calls,
                        "avg_output_tokens": running_avg_tokens,
                        "repair_rate": running_repair_rate,
                        "avg_repairs_per_problem": running_avg_repairs,
                        "state_valid_rate": running_state_valid_rate,
                        "example_correct": int(is_correct),
                        "example_parse_success": int(is_parse_success),
                        "example_abs_error": abs_error,
                        "example_model_calls": result.total_calls,
                        "example_output_tokens": result.total_new_tokens,
                        "example_repairs_used": result.repairs_used,
                        "example_state_valid_rate": result.state_valid_rounds / max(result.total_rounds, 1),
                        "example_complexity": complexity,
                    },
                    step=global_step,
                )
            global_step += 1

    mean_abs_error = abs_error_sum / parse_success if parse_success > 0 else float("nan")
    final_metrics = {
        "accuracy": correct / max(total_examples, 1),
        "mean_abs_error": mean_abs_error,
        "parse_success_rate": parse_success / max(total_examples, 1),
        "avg_model_calls": model_calls_sum / max(total_examples, 1),
        "avg_output_tokens": output_tokens_sum / max(total_examples, 1),
        "repair_rate": repairs_triggered / max(total_examples, 1),
        "avg_repairs_per_problem": repairs_sum / max(total_examples, 1),
        "state_valid_rate": state_valid_rounds / total_rounds if total_rounds > 0 else 0.0,
        "n": total_examples,
    }
    final_metrics.update(compute_accuracy_by_bin(bin_totals))
    return final_metrics


def train_seq2seq(
    model: torch.nn.Module,
    tokenizer,
    train_loader: DataLoader,
    training_cfg: DictConfig,
    cfg: DictConfig,
    log_to_wandb: bool = True,
) -> Dict[str, float]:
    optimizer_name = str(cfg_get(training_cfg, "optimizer", "none")).lower()
    if optimizer_name == "none":
        raise ValueError("Optimizer is set to none; supervised training is disabled.")
    if not model.config.is_encoder_decoder:
        raise ValueError("Supervised training is only supported for encoder-decoder models.")
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_get(training_cfg, "learning_rate", 0.0)),
        weight_decay=float(cfg_get(training_cfg, "weight_decay", 0.0)),
    )
    epochs = max(int(cfg_get(training_cfg, "epochs", 1)), 1)
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg_get(training_cfg, "warmup_steps", 0)),
        num_training_steps=total_steps,
    )
    grad_accum = max(int(cfg_get(training_cfg, "gradient_accumulation_steps", 1)), 1)

    running_loss = 0.0
    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            if step == 0:
                assert batch["input_ids"].shape[0] == batch["labels"].shape[0]
                assert batch["input_ids"].shape == batch["attention_mask"].shape

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=model.config.decoder_start_token_id,
                )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            )
            logits = outputs.logits
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            if any(g is not None for g in grads):
                grad_norm = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads if g is not None))
            else:
                grad_norm = torch.tensor(0.0, device=loss.device)

            loss_to_backprop = loss / grad_accum
            loss_to_backprop.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                assert_gradients(model)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            if log_to_wandb and cfg.wandb.mode != "disabled":
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_grad_norm": grad_norm.item(),
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            global_step += 1

    avg_loss = running_loss / max(global_step, 1)
    model.eval()
    return {"train_loss": avg_loss}


def run_optuna(
    cfg: DictConfig,
    run_cfg: DictConfig,
    optuna_cfg: Optional[DictConfig],
    model_wrapper: ModelWrapper,
    dev_loader: DataLoader,
    tokenizer,
) -> Dict[str, Any]:
    if optuna_cfg is None:
        return {}
    if int(cfg_get(optuna_cfg, "n_trials", 0)) <= 0:
        return {}
    search_spaces = cfg_get(optuna_cfg, "search_spaces", [])
    if not search_spaces:
        return {}

    def suggest_value(trial: optuna.Trial, space: DictConfig):
        dist = str(cfg_get(space, "distribution_type", ""))
        if dist == "int":
            return trial.suggest_int(space.param_name, int(space.low), int(space.high))
        if dist == "loguniform":
            return trial.suggest_float(space.param_name, float(space.low), float(space.high), log=True)
        if dist == "categorical":
            return trial.suggest_categorical(space.param_name, list(space.choices))
        return trial.suggest_float(space.param_name, float(space.low), float(space.high))

    def objective(trial: optuna.Trial) -> float:
        trial_params = {}
        for space in search_spaces:
            trial_params[space.param_name] = suggest_value(trial, space)

        trial_run_cfg = OmegaConf.create(OmegaConf.to_container(run_cfg, resolve=True))
        for key, value in trial_params.items():
            if key == "R_max":
                trial_run_cfg.protocol.rounds.R_max = int(value)
            else:
                trial_run_cfg.protocol[key] = value

        protocol = build_protocol(trial_run_cfg, model_wrapper)
        metrics = evaluate_dataset(
            protocol,
            dev_loader,
            trial_run_cfg.dataset,
            cfg,
            tokenizer,
            log_to_wandb=False,
        )
        return float(metrics["accuracy"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(cfg_get(optuna_cfg, "n_trials", 0)))
    return study.best_params


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = apply_mode_overrides(cfg)
    set_seed(int(cfg.seed))
    run_id = _get_run_id(cfg)

    if cfg.mode != "trial" and cfg.wandb.mode == "disabled":
        raise RuntimeError("WandB must be enabled for full runs.")

    if "run" not in cfg or "run_id" not in cfg.run:
        raise RuntimeError("cfg.run.run_id is required for WandB logging.")

    cache_dir = Path(hydra.utils.to_absolute_path(cfg_get(cfg, "cache_dir", ".cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(hydra.utils.to_absolute_path(cfg.results_dir))
    results_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = resolve_run_cfg(cfg)
    model_cfg = cfg_get(run_cfg, "model", cfg_get(cfg, "model"))
    dataset_cfg = cfg_get(run_cfg, "dataset", cfg_get(cfg, "dataset"))
    training_cfg = cfg_get(run_cfg, "training", cfg_get(cfg, "training"))
    optuna_cfg = cfg_get(run_cfg, "optuna", cfg_get(cfg, "optuna"))

    tokenizer, model = load_model_and_tokenizer(model_cfg, cache_dir)
    preproc = cfg_get(dataset_cfg, "preprocessing", {})
    max_length = int(cfg_get(preproc, "max_length", 512))
    model_wrapper = ModelWrapper(model=model, tokenizer=tokenizer, max_length=max_length)

    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id is missing"
    output_embeddings = model.get_output_embeddings()
    assert output_embeddings is not None, "Model output embeddings missing"
    assert output_embeddings.weight.shape[0] >= len(tokenizer), "Output embeddings mismatch"

    main_dataset, dev_dataset = load_dataset_splits(dataset_cfg, cache_dir)
    if len(main_dataset) == 0:
        raise RuntimeError("Main dataset is empty after preprocessing.")

    if cfg.mode == "trial" and cfg.trial_max_examples:
        max_examples = int(cfg.trial_max_examples)
        main_dataset = main_dataset.select(range(min(len(main_dataset), max_examples)))
        if dev_dataset is not None:
            dev_dataset = dev_dataset.select(range(min(len(dev_dataset), max_examples)))

    batch_size = max(int(cfg_get(training_cfg, "batch_size", 1)), 1)

    dev_loader = None
    if dev_dataset is not None and len(dev_dataset) > 0:
        dev_loader = build_dataloader(dev_dataset, batch_size=batch_size)

    best_params = {}
    if dev_loader is not None:
        best_params = run_optuna(cfg, run_cfg, optuna_cfg, model_wrapper, dev_loader, tokenizer)
        for key, value in best_params.items():
            if key == "R_max":
                run_cfg.protocol.rounds.R_max = int(value)
            else:
                run_cfg.protocol[key] = value

    wandb_run = None
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )

    final_metrics: Dict[str, Any] = {}
    training_mode = str(cfg_get(training_cfg, "mode", "inference_only")).lower()
    epochs = int(cfg_get(training_cfg, "epochs", 0))
    if training_mode == "inference_only" or epochs <= 0:
        protocol = build_protocol(run_cfg, model_wrapper)
        main_loader = build_dataloader(main_dataset, batch_size=batch_size)
        final_metrics = evaluate_dataset(
            protocol,
            main_loader,
            dataset_cfg,
            cfg,
            tokenizer,
            log_to_wandb=True,
            max_batches=cfg.trial_max_batches,
        )
    else:
        train_loader = build_seq2seq_dataloader(
            main_dataset,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
        )
        final_metrics = train_seq2seq(model, tokenizer, train_loader, training_cfg, cfg, log_to_wandb=True)

        protocol = build_protocol(run_cfg, model_wrapper)
        main_loader = build_dataloader(main_dataset, batch_size=batch_size)
        eval_metrics = evaluate_dataset(
            protocol,
            main_loader,
            dataset_cfg,
            cfg,
            tokenizer,
            log_to_wandb=True,
            max_batches=cfg.trial_max_batches,
        )
        final_metrics.update(eval_metrics)

    if cfg.wandb.mode != "disabled" and wandb_run is not None:
        for key, value in final_metrics.items():
            wandb.summary[key] = value
        if best_params:
            wandb.summary["optuna_best_params"] = best_params
        print(wandb_run.url)
        wandb.finish()
    else:
        print("WandB disabled; run completed.")


if __name__ == "__main__":
    main()
