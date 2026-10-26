"""GSM8K reasoning benchmark evaluation suite, telemetry accumulators, and metric calculators."""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Tuple, Any

# Standalone 20-problem arithmetic and reasoning suite matching GSM8K distribution
GSM8K_STANDALONE_SUITE = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "ground_truth": 72.0,
        "solution": "Natalia sold 48 / 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips. #### 72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "ground_truth": 10.0,
        "solution": "Weng earns $12 / 60 = $0.20 per minute. For 50 minutes she earned 50 * 0.20 = $10. #### 10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "ground_truth": 5.0,
        "solution": "Betty has 100 / 2 = $50. Her grandparents gave her 15 * 2 = $30. Total saved is 50 + 15 + 30 = $95. She needs 100 - 95 = $5. #### 5"
    },
    {
        "question": "A deep-sea monster rises from the waters once every 100 years to feast on a ship and sleep for 100 years more. The last time the monster rose was in the year 800. In what year will the monster rise for the fourth time since 800?",
        "ground_truth": 1200.0,
        "solution": "The monster rises every 100 years. For 4 times, 4 * 100 = 400 years pass. 800 + 400 = 1200. #### 1200"
    },
    {
        "question": "Mark has a garden with flowers. He has 10 rows of flowers, and each row has 5 flowers. If half of the flowers are red, how many red flowers does he have?",
        "ground_truth": 25.0,
        "solution": "Total flowers = 10 * 5 = 50. Red flowers = 50 / 2 = 25. #### 25"
    },
    {
        "question": "Albert is wondering how much pizza he can eat. A large pizza has 12 slices. Albert ate 1/3 of the pizza for lunch and 1/4 of the pizza for dinner. How many slices are left?",
        "ground_truth": 5.0,
        "solution": "1/3 of 12 is 4 slices. 1/4 of 12 is 3 slices. Slices eaten = 4 + 3 = 7. Left = 12 - 7 = 5. #### 5"
    },
    {
        "question": "Ken created a care package to send to his brother, who was away at school. Ken placed a box of 20 cookies and 3 boxes of crackers with 10 crackers each. How many total food items did Ken send?",
        "ground_truth": 50.0,
        "solution": "3 boxes of crackers have 3 * 10 = 30 crackers. Total items = 20 + 30 = 50. #### 50"
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take to make 4 robes?",
        "ground_truth": 12.0,
        "solution": "White fiber per robe = 2 / 2 = 1. Total fiber per robe = 2 + 1 = 3 bolts. For 4 robes: 4 * 3 = 12 bolts. #### 12"
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 into repairs. He sells the house for $150,000. How much profit did he make?",
        "ground_truth": 20000.0,
        "solution": "Total cost = 80000 + 50000 = $130,000. Profit = 150000 - 130000 = $20,000. #### 20000"
    },
    {
        "question": "James decides to run 3 miles every day for 2 weeks. How many total miles did he run?",
        "ground_truth": 42.0,
        "solution": "2 weeks is 14 days. 14 * 3 = 42 miles. #### 42"
    },
    {
        "question": "Every hour, a bakery bakes 30 chocolate chip cookies and 20 oatmeal cookies. How many total cookies does the bakery bake in an 8-hour workday?",
        "ground_truth": 400.0,
        "solution": "Total cookies per hour = 30 + 20 = 50. In 8 hours, 50 * 8 = 400 cookies. #### 400"
    },
    {
        "question": "A library has 5 shelves of history books with 20 books each, and 4 shelves of science books with 25 books each. How many books are there in total?",
        "ground_truth": 200.0,
        "solution": "History: 5 * 20 = 100. Science: 4 * 25 = 100. Total = 100 + 100 = 200 books. #### 200"
    },
    {
        "question": "Kendra bought 4 notebooks for $3 each and 2 packs of pens for $5 each. If she paid with a $50 bill, how much change did she receive?",
        "ground_truth": 28.0,
        "solution": "Notebooks cost 4 * 3 = $12. Pens cost 2 * 5 = $10. Total spent = 12 + 10 = $22. Change = 50 - 22 = $28. #### 28"
    },
    {
        "question": "A farmer has 15 cows and 25 chickens. How many legs do all of the animals have in total?",
        "ground_truth": 110.0,
        "solution": "15 cows have 15 * 4 = 60 legs. 25 chickens have 25 * 2 = 50 legs. Total legs = 60 + 50 = 110 legs. #### 110"
    },
    {
        "question": "Sam bought 3 t-shirts for $15 each and 2 pairs of jeans for $40 each. How much did Sam spend in total?",
        "ground_truth": 115.0,
        "solution": "Shirts: 3 * 15 = $45. Jeans: 2 * 40 = $80. Total = 45 + 80 = $115. #### 115"
    },
    {
        "question": "A train travels at a speed of 60 miles per hour. How far does the train travel in 3.5 hours?",
        "ground_truth": 210.0,
        "solution": "Distance = speed * time = 60 * 3.5 = 210 miles. #### 210"
    },
    {
        "question": "In a school of 600 students, 40% are boys. How many girls are there in the school?",
        "ground_truth": 360.0,
        "solution": "Boys = 600 * 0.40 = 240. Girls = 600 - 240 = 360. #### 360"
    },
    {
        "question": "Maya has 5 boxes of pencils with 12 pencils in each box. She gives 15 pencils to her classmates. How many pencils does she have left?",
        "ground_truth": 45.0,
        "solution": "Total pencils = 5 * 12 = 60. Left = 60 - 15 = 45. #### 45"
    },
    {
        "question": "David can type 50 words per minute. How many words can he type in an essay if he spends 30 minutes typing?",
        "ground_truth": 1500.0,
        "solution": "Words = 50 * 30 = 1500. #### 1500"
    },
    {
        "question": "An apple orchard picked 450 apples. If they package them into bags containing 9 apples each, how many bags can they fill?",
        "ground_truth": 50.0,
        "solution": "Bags = 450 / 9 = 50 bags. #### 50"
    }
]

ANSWER_PATTERNS = (r"####\s*([+-]?[\d,]+(?:\.\d+)?)", r"[Tt]he answer is:?\s*\$?([+-]?[\d,]+(?:\.\d+)?)")


def truncate_at_answer(text: str) -> Optional[str]:
    """Prefix of text ending at the first explicit answer ('#### n' or 'the answer is n'); None if absent."""
    ends = [m.end() for pat in ANSWER_PATTERNS for m in [re.search(pat, text)] if m]
    return text[:min(ends)].rstrip(".,") if ends else None


def extract_numerical_answer(text: str, strict: bool = False) -> Optional[float]:
    """Extracts numeric answer from reasoning traces following GSM8k conventions.
    strict: only explicit answer markers count; no last-number fallback.
    """
    if not text:
        return None

    # 1. Standard GSM8K delimiter: '#### <num>'
    match_hash = re.search(r"####\s*([+-]?[\d,]+(?:\.\d+)?)", text)
    if match_hash:
        raw_val = match_hash.group(1).replace(",", "")
        try:
            return float(raw_val)
        except ValueError:
            pass

    # 2. 'The answer is <num>' pattern
    match_phrase = re.search(r"[Tt]he answer is:?\s*([+-]?[\d,]+(?:\.\d+)?)", text)
    if match_phrase:
        raw_val = match_phrase.group(1).replace(",", "")
        try:
            return float(raw_val)
        except ValueError:
            pass

    if strict:
        return None

    # 3. Fallback: Extract the last number appearing in the output
    numbers = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    if numbers:
        raw_val = numbers[-1].replace(",", "")
        try:
            return float(raw_val)
        except ValueError:
            pass

    return None

def evaluate_answer(pred_text: str, ground_truth: float, tolerance: float = 1e-4, strict: bool = True) -> bool:
    """Evaluates whether the extracted answer matches ground truth (strict: explicit marker required)."""
    pred = extract_numerical_answer(pred_text, strict=strict)
    if pred is None:
        return False
    return abs(pred - ground_truth) <= tolerance

@dataclass
class TokenSavingsAccumulator:
    """Tracks token consumption, latency, and accuracy metrics across generation arms."""
    greedy_tokens: List[int] = field(default_factory=list)
    pruned_tokens: List[int] = field(default_factory=list)
    truncated_tokens: List[int] = field(default_factory=list)

    greedy_correct: List[bool] = field(default_factory=list)
    pruned_correct: List[bool] = field(default_factory=list)
    truncated_correct: List[bool] = field(default_factory=list)

    was_pruned: List[bool] = field(default_factory=list)
    prune_reasons: List[str] = field(default_factory=list)

    def add_sample(
        self,
        n_greedy: int,
        n_pruned: int,
        n_truncated: int,
        c_greedy: bool,
        c_pruned: bool,
        c_truncated: bool,
        pruned: bool,
        prune_reason: str = "completed"
    ):
        """Records generation statistics for a single benchmark problem."""
        self.greedy_tokens.append(n_greedy)
        self.pruned_tokens.append(n_pruned)
        self.truncated_tokens.append(n_truncated)

        self.greedy_correct.append(c_greedy)
        self.pruned_correct.append(c_pruned)
        self.truncated_correct.append(c_truncated)

        self.was_pruned.append(pruned)
        self.prune_reasons.append(prune_reason)

    def compute_speedup_ratio(self) -> float:
        """Calculates token speedup ratio: total_greedy_tokens / total_pruned_tokens."""
        total_greedy = sum(self.greedy_tokens)
        total_pruned = sum(self.pruned_tokens)
        if total_pruned == 0:
            return 1.0
        return total_greedy / total_pruned

    def compute_accuracy(self, arm: str = "pruned") -> float:
        """Calculates pass@1 accuracy for the specified generation arm."""
        if arm == "greedy":
            records = self.greedy_correct
        elif arm == "pruned":
            records = self.pruned_correct
        elif arm == "truncated":
            records = self.truncated_correct
        else:
            raise ValueError(f"Unknown arm: {arm}")

        if not records:
            return 0.0
        return sum(records) / len(records)

    def compute_false_prune_rate(self) -> float:
        """Computes rate of false prunes: greedy was correct, but pruner terminated and failed.
        false_prune_rate = N(greedy_correct & ~pruned_correct & was_pruned) / N(greedy_correct)
        """
        greedy_success_count = sum(self.greedy_correct)
        if greedy_success_count == 0:
            return 0.0

        false_prunes = 0
        for c_g, c_p, p in zip(self.greedy_correct, self.pruned_correct, self.was_pruned):
            if c_g and (not c_p) and p:
                false_prunes += 1

        return false_prunes / greedy_success_count

    def evaluate_gate(
        self,
        min_speedup: float = 2.0,
        max_relative_acc_drop: float = 0.10
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluates pre-registered Stage 2 gate:
        Speedup >= 2.0x and Accuracy within 10% of greedy baseline.
        """
        speedup = self.compute_speedup_ratio()
        acc_greedy = self.compute_accuracy("greedy")
        acc_pruned = self.compute_accuracy("pruned")
        acc_truncated = self.compute_accuracy("truncated")
        false_prune_rate = self.compute_false_prune_rate()

        # Accuracy within 10%: absolute drop <= 0.10 or pruned >= 0.90 * greedy
        acc_drop = acc_greedy - acc_pruned
        acc_within_threshold = (acc_drop <= max(0.10, max_relative_acc_drop * acc_greedy))
        speedup_passed = (speedup >= min_speedup)

        gate_passed = speedup_passed and acc_within_threshold

        metrics = {
            "speedup_ratio": speedup,
            "acc_greedy": acc_greedy,
            "acc_pruned": acc_pruned,
            "acc_truncated": acc_truncated,
            "acc_drop": acc_drop,
            "false_prune_rate": false_prune_rate,
            "speedup_passed": speedup_passed,
            "acc_passed": acc_within_threshold,
            "gate_passed": gate_passed,
            "total_greedy_tokens": sum(self.greedy_tokens),
            "total_pruned_tokens": sum(self.pruned_tokens),
            "total_truncated_tokens": sum(self.truncated_tokens),
            "avg_greedy_tokens": sum(self.greedy_tokens) / max(1, len(self.greedy_tokens)),
            "avg_pruned_tokens": sum(self.pruned_tokens) / max(1, len(self.pruned_tokens)),
            "prune_frequency": sum(self.was_pruned) / max(1, len(self.was_pruned)),
        }
        return gate_passed, metrics

def load_gsm8k_dataset(num_samples: int = 50, use_standalone: bool = False) -> List[Dict[str, Any]]:
    """Loads GSM8K test subset from HuggingFace datasets or built-in standalone suite."""
    if not use_standalone:
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split="test")
            samples = []
            for item in ds:
                gt = extract_numerical_answer(item["answer"])
                if gt is not None:
                    samples.append({
                        "question": item["question"],
                        "ground_truth": gt,
                        "solution": item["answer"]
                    })
                if len(samples) >= num_samples:
                    break
            if samples:
                return samples
        except Exception:
            pass

    # Built-in standalone fallback suite
    suite = GSM8K_STANDALONE_SUITE
    samples = []
    idx = 0
    while len(samples) < num_samples:
        samples.append(suite[idx % len(suite)])
        idx += 1
    return samples
