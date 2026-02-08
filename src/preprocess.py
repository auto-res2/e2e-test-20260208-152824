import re
from dataclasses import dataclass
from typing import Optional, Tuple

from datasets import load_dataset

NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def normalize_numeric(text: str) -> str:
    return str(text).replace(",", "").strip()


def parse_numeric(text: str) -> Optional[float]:
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


def extract_gsm8k_gold(answer: str) -> Optional[float]:
    if answer is None:
        return None
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", answer)
    if match:
        return parse_numeric(match.group(1))
    return parse_numeric(answer)


def extract_svamp_gold(example) -> Optional[float]:
    return parse_numeric(example.get("Answer"))


def extract_multi_arith_gold(example) -> Optional[float]:
    return parse_numeric(
        example.get("final_ans")
        or example.get("answer")
        or example.get("lSolutions")
        or example.get("lSolution")
    )


def count_numbers(text: str) -> int:
    if not text:
        return 0
    return len(NUM_RE.findall(text))


@dataclass
class ReasoningExample:
    example_id: str
    question: str
    gold_value: float
    gold_text: str
    complexity: int
    metadata: dict


class ReasoningDataset:
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def select(self, indices):
        if isinstance(indices, range):
            indices = list(indices)
        return ReasoningDataset([self.examples[i] for i in indices])


def get_question_answer(dataset_name: str, example) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    name = dataset_name.lower()
    if "gsm8k" in name:
        question = example.get("question")
        answer_text = example.get("answer")
        gold = extract_gsm8k_gold(answer_text)
        return question, gold, answer_text
    if "svamp" in name:
        question = f"{example.get('Body', '')} {example.get('Question', '')}".strip()
        answer_text = str(example.get("Answer"))
        gold = extract_svamp_gold(example)
        return question, gold, answer_text
    if "multi_arith" in name or "multiarith" in name:
        question = example.get("question") or example.get("sQuestion") or example.get("Question")
        answer_text = str(
            example.get("final_ans")
            or example.get("answer")
            or example.get("lSolutions")
            or example.get("lSolution")
        )
        gold = extract_multi_arith_gold(example)
        return question, gold, answer_text
    question = example.get("question") or example.get("Question")
    answer_text = str(example.get("answer") or example.get("Answer"))
    gold = parse_numeric(answer_text)
    return question, gold, answer_text


def build_examples(dataset_name: str, raw_dataset, normalize: bool) -> ReasoningDataset:
    examples = []
    if raw_dataset is None:
        return ReasoningDataset(examples)
    for idx, ex in enumerate(raw_dataset):
        question, gold, answer_text = get_question_answer(dataset_name, ex)
        if question is None or gold is None:
            continue
        if normalize:
            question = question.strip()
        examples.append(
            ReasoningExample(
                example_id=str(idx),
                question=question,
                gold_value=gold,
                gold_text=str(answer_text),
                complexity=count_numbers(question),
                metadata={"raw": ex},
            )
        )
    return ReasoningDataset(examples)


def load_split(name: str, config: Optional[str], split: str, cache_dir: str):
    if config:
        return load_dataset(name, config, split=split, cache_dir=cache_dir)
    return load_dataset(name, split=split, cache_dir=cache_dir)


def load_dataset_splits(cfg_dataset, cache_dir):
    dataset_name = cfg_dataset.name
    config = cfg_dataset.get("config")
    split = cfg_dataset.split
    dev_split = cfg_dataset.get("dev_split")

    candidates = [dataset_name]
    name_lower = dataset_name.lower()
    if dataset_name.startswith("openai/"):
        candidates.append(dataset_name.split("/", 1)[1])
    if "gsm8k" in name_lower:
        candidates.append("gsm8k")
    if "svamp" in name_lower:
        candidates.append("svamp")
    if "multi_arith" in name_lower or "multiarith" in name_lower:
        candidates.extend(["multi_arith", "multiarith", "lighteval/MultiArith"])

    last_error = None
    main_split = None
    resolved_name = dataset_name
    for candidate in candidates:
        try:
            main_split = load_split(candidate, config, split, str(cache_dir))
            resolved_name = candidate
            break
        except Exception as exc:
            last_error = exc
    if main_split is None:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {last_error}")

    dev_dataset = None
    if dev_split:
        try:
            dev_dataset = load_split(resolved_name, config, dev_split, str(cache_dir))
        except Exception:
            dev_dataset = None

    normalize = bool(cfg_dataset.preprocessing.get("normalize_numeric", True))
    main_dataset = build_examples(resolved_name, main_split, normalize)
    dev_dataset = build_examples(resolved_name, dev_dataset, normalize) if dev_dataset is not None else None
    return main_dataset, dev_dataset
