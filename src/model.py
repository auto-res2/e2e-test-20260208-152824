import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def cfg_get(cfg_obj: Any, key: str, default: Any = None) -> Any:
    if cfg_obj is None:
        return default
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    if hasattr(cfg_obj, "get"):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


@dataclass
class ProtocolResult:
    prediction_text: str
    prediction_value: Optional[float]
    total_calls: int
    total_new_tokens: int
    repairs_used: int
    state_valid_rounds: int
    total_rounds: int


def normalize_numeric(text: str) -> str:
    return str(text).replace(",", "").strip()


def parse_numeric_value(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    clean = normalize_numeric(str(text))
    nums = NUM_RE.findall(clean)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def numeric_equal(pred: float, gold: float) -> bool:
    if pred is None or gold is None:
        return False
    if not (math.isfinite(pred) and math.isfinite(gold)):
        return False
    tol = max(1e-6, 1e-4 * max(1.0, abs(gold)))
    return abs(pred - gold) <= tol


def count_numbers(text: str) -> int:
    return len(NUM_RE.findall(text or ""))


def plan_rounds(question: str, rounds_cfg) -> Tuple[int, int]:
    heuristic = str(cfg_get(rounds_cfg, "heuristic", "number_count"))
    n = count_numbers(question)
    if heuristic == "number_count":
        if n <= 2:
            r, k = 1, 6
        elif n <= 4:
            r, k = 2, 5
        else:
            r, k = 3, 4
    else:
        r, k = int(cfg_get(rounds_cfg, "R_max", 3)), 5
    r_max = int(cfg_get(rounds_cfg, "R_max", r))
    r = max(1, min(r, r_max))
    return r, k


def prompt_direct(question: str) -> str:
    return f"Solve the problem. Reply with one line: FINAL: <number>\nQuestion: {question}\nFINAL:"


def prompt_cot(question: str) -> str:
    return (
        "Solve the problem. Show reasoning then answer.\n"
        f"Question: {question}\n"
        "Let's think step by step.\nFINAL:"
    )


def prompt_sr_round(question: str, state: str, r: int, R: int, k: int) -> str:
    return (
        "You solve in rounds and carry only STATE forward.\n"
        f"Round {r}/{R}.\n"
        f"Question: {question}\n"
        f"STATE: {state}\n\n"
        f"Do up to {k} short steps to update the state. Then output:\n"
        "STATE: {key: value, ...} (compact)\n"
        "If you are sure, output FINAL: <number> instead.\n"
    )


def prompt_ecsr_round(
    question: str,
    state: str,
    r: int,
    R: int,
    k: int,
    invariant_count: int,
) -> str:
    return (
        "You solve in rounds and carry only a compact STATE forward.\n"
        "Before continuing, you must make the STATE self-checking.\n"
        f"Round {r}/{R}.\n"
        f"Question: {question}\n"
        f"STATE: {state}\n\n"
        f"Do up to {k} short steps to update the state (steps will be discarded).\n"
        "Then output either:\n"
        "FINAL: <number>\n"
        "OR exactly this format:\n"
        "STATE: {key: value, ...}   (<=10 keys; use numbers when possible)\n"
        "INVS:\n"
        "k1 == <expr using keys and + - * / ()>\n"
        "k2 == <expr using keys and + - * / ()>\n"
        f"(Provide {invariant_count} invariants; each must reference at least one key)\n"
    )


def prompt_repair(question: str, bad_state_text: str, errors: str) -> str:
    return (
        "Repair the STATE so all invariants are true. Output only STATE and INVS.\n"
        f"Question: {question}\n\n"
        f"BAD_OUTPUT:\n{bad_state_text}\n\n"
        f"ERRORS:\n{errors}\n\n"
        "STATE:"
    )


def prompt_finalize(question: str, state: str) -> str:
    return (
        "Use only the STATE to answer. Output exactly one line: FINAL: <number>\n"
        f"Question: {question}\nSTATE: {state}\nFINAL:"
    )


def extract_final(text: str) -> Optional[str]:
    match = re.search(r"FINAL:\s*([-+]?\d*\.?\d+)", text)
    if match:
        return match.group(1)
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def extract_state_block(text: str) -> Optional[str]:
    if "STATE:" not in text:
        return None
    idx = text.find("STATE:")
    sub = text[idx + len("STATE:") :]
    start = sub.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(sub[start:]):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return sub[start : start + i + 1].strip()
    return None


def extract_invs(text: str) -> List[str]:
    if "INVS" not in text:
        return []
    part = text.split("INVS:", 1)[1]
    invs = []
    for line in part.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("STATE") or line.startswith("FINAL"):
            break
        if "==" in line:
            invs.append(line)
    return invs[:4]


def parse_state(state_str: str) -> Dict[str, Any]:
    obj = ast.literal_eval(state_str)
    if not isinstance(obj, dict):
        raise ValueError("STATE is not a dict")
    if len(obj) > 10:
        raise ValueError("STATE exceeds 10 keys")
    return obj


_ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Load,
    ast.Name,
    ast.Constant,
}


def safe_eval(expr: str, env: Dict[str, float]) -> float:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(f"Disallowed node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in env:
            raise ValueError(f"Unknown key: {node.id}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Non-numeric constant")
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)


def validate_state(state: Dict[str, Any], invs: List[str], tol: float) -> Tuple[bool, List[str]]:
    if not invs:
        return False, ["missing INVS"]
    errors = []
    env: Dict[str, float] = {}
    for k, v in state.items():
        if isinstance(v, (int, float)):
            env[k] = float(v)
        else:
            try:
                env[k] = float(str(v))
            except Exception:
                continue
    for inv in invs:
        try:
            lhs, rhs = [x.strip() for x in inv.split("==", 1)]
            if lhs not in env:
                errors.append(f"lhs not numeric or missing: {lhs}")
                continue
            rhs_val = safe_eval(rhs, env)
            if not (math.isfinite(rhs_val) and math.isfinite(env[lhs])):
                errors.append(f"non-finite in {inv}")
                continue
            if abs(env[lhs] - float(rhs_val)) > tol:
                errors.append(f"fails: {inv} (lhs={env[lhs]}, rhs={rhs_val})")
        except Exception as exc:
            errors.append(f"bad inv '{inv}': {exc}")
    return len(errors) == 0, errors


def _resolve_device(model: torch.nn.Module) -> torch.device:
    try:
        device = next(model.parameters()).device
        if device.type != "meta":
            return device
    except StopIteration:
        pass
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first = next(iter(model.hf_device_map.values()))
        if isinstance(first, int):
            return torch.device(f"cuda:{first}" if torch.cuda.is_available() else "cpu")
        return torch.device(str(first))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelWrapper:
    def __init__(self, model, tokenizer, max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = _resolve_device(model)

    def generate(
        self, prompt: str, max_new_tokens: int, do_sample: bool, num_beams: int
    ) -> Tuple[str, int]:
        max_len = self.max_length or self.tokenizer.model_max_length
        if self.tokenizer.model_max_length and self.tokenizer.model_max_length < max_len:
            max_len = self.tokenizer.model_max_length
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                num_beams=int(num_beams),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        if self.model.config.is_encoder_decoder:
            new_tokens = output.shape[-1]
        else:
            new_tokens = max(output.shape[-1] - inputs["input_ids"].shape[-1], 0)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text, int(new_tokens)


def load_model_and_tokenizer(cfg_model, cache_dir: Path):
    config = AutoConfig.from_pretrained(cfg_model.name, cache_dir=str(cache_dir), trust_remote_code=True)
    dtype = None
    dtype_name = str(cfg_model.get("dtype", "")) if cfg_model is not None else ""
    if dtype_name:
        if dtype_name.lower() in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif dtype_name.lower() in {"fp16", "float16"}:
            dtype = torch.float16
        elif dtype_name.lower() in {"fp32", "float32"}:
            dtype = torch.float32
    model_cls = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        cfg_model.name,
        cache_dir=str(cache_dir),
        torch_dtype=dtype,
        device_map=cfg_model.get("device_map"),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg_model.name, cache_dir=str(cache_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    tokenizer.padding_side = "right" if config.is_encoder_decoder else "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    if config.is_encoder_decoder and model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model


class BaseProtocol:
    def __init__(self, model_wrapper: ModelWrapper, protocol_cfg, decoding_cfg):
        self.model_wrapper = model_wrapper
        self.protocol_cfg = protocol_cfg
        self.decoding_cfg = decoding_cfg or {}

    def _get_decoding_param(self, name: str, default: Any) -> Any:
        return cfg_get(self.decoding_cfg, name, default)

    def _generate(self, prompt: str, max_new_tokens: int) -> Tuple[str, int]:
        return self.model_wrapper.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=self._get_decoding_param("do_sample", False),
            num_beams=self._get_decoding_param("num_beams", 1),
        )


class DirectProtocol(BaseProtocol):
    def run(self, question: str) -> ProtocolResult:
        max_new = int(cfg_get(self.protocol_cfg, "max_new_tokens_final", 64))
        text, tokens = self._generate(prompt_direct(question), max_new_tokens=max_new)
        pred_str = extract_final(text)
        pred_val = parse_numeric_value(pred_str) if pred_str is not None else None
        return ProtocolResult(text, pred_val, 1, tokens, 0, 0, 0)


class CoTProtocol(BaseProtocol):
    def run(self, question: str) -> ProtocolResult:
        max_new = int(cfg_get(self.protocol_cfg, "max_new_tokens_final", 64))
        text, tokens = self._generate(prompt_cot(question), max_new_tokens=max_new)
        pred_str = extract_final(text)
        pred_val = parse_numeric_value(pred_str) if pred_str is not None else None
        return ProtocolResult(text, pred_val, 1, tokens, 0, 0, 0)


class SRResetProtocol(BaseProtocol):
    def run(self, question: str) -> ProtocolResult:
        state = "{}"
        total_calls = 0
        total_tokens = 0
        rounds, k = plan_rounds(question, self.protocol_cfg.rounds)
        pred_val = None
        final_text = ""
        rounds_attempted = 0
        max_new_round = int(cfg_get(self.protocol_cfg, "max_new_tokens_round", 192))
        max_new_final = int(cfg_get(self.protocol_cfg, "max_new_tokens_final", 64))
        for r in range(1, rounds + 1):
            rounds_attempted += 1
            prompt = prompt_sr_round(question, state, r, rounds, k)
            text, tokens = self._generate(prompt, max_new_tokens=max_new_round)
            total_calls += 1
            total_tokens += tokens
            final_text = text
            pred_str = extract_final(text)
            if "FINAL:" in text and pred_str is not None:
                pred_val = parse_numeric_value(pred_str)
                return ProtocolResult(text, pred_val, total_calls, total_tokens, 0, 0, rounds_attempted)
            state_block = extract_state_block(text)
            if state_block:
                state = state_block
        if pred_val is None:
            final_prompt = prompt_finalize(question, state)
            text, tokens = self._generate(final_prompt, max_new_tokens=max_new_final)
            total_calls += 1
            total_tokens += tokens
            final_text = text
            pred_str = extract_final(text)
            pred_val = parse_numeric_value(pred_str) if pred_str is not None else None
        return ProtocolResult(final_text, pred_val, total_calls, total_tokens, 0, 0, rounds_attempted)


class ECSRProtocol(BaseProtocol):
    def run(self, question: str) -> ProtocolResult:
        state = "{}"
        total_calls = 0
        total_tokens = 0
        repairs_used = 0
        valid_rounds = 0
        rounds, k = plan_rounds(question, self.protocol_cfg.rounds)
        pred_val = None
        final_text = ""
        rounds_attempted = 0

        invariant_count = int(cfg_get(self.protocol_cfg, "invariant_count", 3))
        invariant_count = min(max(invariant_count, 2), 4)
        max_repairs = max(int(cfg_get(self.protocol_cfg, "max_repairs", 0)), 0)
        tol = float(cfg_get(self.protocol_cfg, "tol", 1e-6))
        max_new_round = int(cfg_get(self.protocol_cfg, "max_new_tokens_round", 192))
        max_new_repair = int(cfg_get(self.protocol_cfg, "max_new_tokens_repair", 160))
        max_new_final = int(cfg_get(self.protocol_cfg, "max_new_tokens_final", 64))

        for r in range(1, rounds + 1):
            rounds_attempted += 1
            prompt = prompt_ecsr_round(question, state, r, rounds, k, invariant_count)
            text, tokens = self._generate(prompt, max_new_tokens=max_new_round)
            total_calls += 1
            total_tokens += tokens
            final_text = text
            pred_str = extract_final(text)
            if "FINAL:" in text and pred_str is not None:
                pred_val = parse_numeric_value(pred_str)
                return ProtocolResult(text, pred_val, total_calls, total_tokens, repairs_used, valid_rounds, rounds_attempted)

            state_block = extract_state_block(text)
            invs = extract_invs(text)
            cur_text = text

            valid_state = False
            for attempt in range(max_repairs + 1):
                try:
                    if not state_block:
                        raise ValueError("missing STATE")
                    st = parse_state(state_block)
                    ok, errs = validate_state(st, invs, tol)
                    if ok:
                        if attempt == 0:
                            valid_rounds += 1
                        state = state_block
                        valid_state = True
                        break
                    raise ValueError("; ".join(errs))
                except Exception as exc:
                    if attempt >= max_repairs:
                        break
                    repairs_used += 1
                    rep_prompt = prompt_repair(question, cur_text, str(exc))
                    rep_text, rep_tokens = self._generate(rep_prompt, max_new_tokens=max_new_repair)
                    total_calls += 1
                    total_tokens += rep_tokens
                    cur_text = rep_text
                    state_block = extract_state_block(rep_text)
                    invs = extract_invs(rep_text)

            if not valid_state:
                state = state

        if pred_val is None:
            final_prompt = prompt_finalize(question, state)
            text, tokens = self._generate(final_prompt, max_new_tokens=max_new_final)
            total_calls += 1
            total_tokens += tokens
            final_text = text
            pred_str = extract_final(text)
            pred_val = parse_numeric_value(pred_str) if pred_str is not None else None

        return ProtocolResult(final_text, pred_val, total_calls, total_tokens, repairs_used, valid_rounds, rounds_attempted)
