"""Unit tests for GSM8K benchmark evaluation, answer parsing, and token savings accumulator."""

from energy_search.benchmarks import (
    extract_numerical_answer,
    evaluate_answer,
    TokenSavingsAccumulator,
    load_gsm8k_dataset,
    GSM8K_STANDALONE_SUITE,
)

def test_extract_numerical_answer_patterns():
    # Pattern 1: Standard GSM8K delimiter #### <num>
    ans1 = extract_numerical_answer("Therefore, the total number of items is 48 + 24 = 72. #### 72")
    assert ans1 == 72.0

    # Pattern 2: Numbers with commas
    ans2 = extract_numerical_answer("Josh made a total profit of $20,000. #### 20,000")
    assert ans2 == 20000.0

    # Pattern 3: Phrase 'The answer is <num>'
    ans3 = extract_numerical_answer("After subtracting expenses, the answer is: 115")
    assert ans3 == 115.0

    # Pattern 4: Trailing number fallback
    ans4 = extract_numerical_answer("In 3.5 hours, the train travels 210 miles.")
    assert ans4 == 210.0

    # Pattern 5: Non-numeric returns None
    ans5 = extract_numerical_answer("I cannot determine the answer to this question.")
    assert ans5 is None

def test_evaluate_answer():
    assert evaluate_answer("The result is #### 42", 42.0)
    assert evaluate_answer("Total is 100.0001", 100.0, tolerance=1e-3)
    assert not evaluate_answer("The result is #### 43", 42.0)
    assert not evaluate_answer("No number found", 10.0)

def test_token_savings_accumulator_metrics():
    acc = TokenSavingsAccumulator()

    # Sample 1: Concise (Greedy 30 tokens, Pruned 30 tokens, Truncated 30 tokens; all correct)
    acc.add_sample(n_greedy=30, n_pruned=30, n_truncated=30,
                   c_greedy=True, c_pruned=True, c_truncated=True,
                   pruned=False)

    # Sample 2: Looping problem (Greedy 200 tokens, Pruned 40 tokens, Truncated 40 tokens; Greedy fail, Pruned fail, Trunc fail)
    acc.add_sample(n_greedy=200, n_pruned=40, n_truncated=40,
                   c_greedy=False, c_pruned=False, c_truncated=False,
                   pruned=True, prune_reason="cognitive_freeze_loop")

    # Sample 3: Long reasoning (Greedy 60 tokens, Pruned 60 tokens, Truncated 40 tokens; Greedy correct, Pruned correct, Trunc fail)
    acc.add_sample(n_greedy=60, n_pruned=60, n_truncated=40,
                   c_greedy=True, c_pruned=True, c_truncated=False,
                   pruned=False)

    # Speedup = (30 + 200 + 60) / (30 + 40 + 60) = 290 / 130 = 2.23x
    speedup = acc.compute_speedup_ratio()
    assert abs(speedup - (290 / 130)) < 1e-4

    # Accuracies
    assert acc.compute_accuracy("greedy") == 2 / 3
    assert acc.compute_accuracy("pruned") == 2 / 3
    assert acc.compute_accuracy("truncated") == 1 / 3

    # False prune rate (Greedy correct, Pruned wrong, was pruned) -> 0 in this set
    assert acc.compute_false_prune_rate() == 0.0

    # Gate evaluation
    gate_passed, metrics = acc.evaluate_gate(min_speedup=2.0, max_relative_acc_drop=0.10)
    assert gate_passed
    assert metrics["speedup_passed"]
    assert metrics["acc_passed"]

def test_false_prune_detection():
    acc = TokenSavingsAccumulator()
    # Greedy correct, but pruner killed prematurely causing incorrect answer
    acc.add_sample(n_greedy=80, n_pruned=20, n_truncated=20,
                   c_greedy=True, c_pruned=False, c_truncated=False,
                   pruned=True, prune_reason="chaotic_divergence")
    # Greedy correct, pruned correct
    acc.add_sample(n_greedy=40, n_pruned=40, n_truncated=40,
                   c_greedy=True, c_pruned=True, c_truncated=True,
                   pruned=False)

    # 1 false prune out of 2 greedy successes -> 50% false prune rate
    assert abs(acc.compute_false_prune_rate() - 0.50) < 1e-4

def test_standalone_gsm8k_suite():
    assert len(GSM8K_STANDALONE_SUITE) == 20
    for problem in GSM8K_STANDALONE_SUITE:
        assert "question" in problem
        assert "ground_truth" in problem
        assert "solution" in problem
        # Verify solution parses to ground truth
        assert evaluate_answer(problem["solution"], problem["ground_truth"])

    # Test loading with cycling
    dataset_35 = load_gsm8k_dataset(num_samples=35, use_standalone=True)
    assert len(dataset_35) == 35
