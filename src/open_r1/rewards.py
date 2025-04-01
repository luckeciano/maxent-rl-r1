"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from typing import Dict
import warnings
import torch

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)
    try:
        rewards = run_async_from_sync(scripts, language)

    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str) -> list[float]:
    # Create the sandbox by hand, currently there's no context manager for this version
    sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(sbx, script, language) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    # Kill the sandbox after all the tasks are complete
    await sbx.kill()

    return rewards


async def run_script(sbx: AsyncSandbox, script: str, language: str) -> float:
    execution = await sbx.run_code(script, language=language)
    try:
        return float(execution.text)
    except (TypeError, ValueError):
        return 0.0

def get_token_entropy_reward(reduction: str = "sum"):
    """
    Get a token entropy reward function that can be used to calculate the entropy of the tokens in a completion.

    Args:
        reduction: The reduction method to use for the entropy reward.
    """       

    def token_entropy_reward(completions, logprobs=None, **kwargs) -> list[float]:
        """Calculate entropy reward using token log probabilities.
        
        Args:
            completions: List of model completions 
            logprobs: List of token log probabilities for each completion (shape: (B, T))
            **kwargs: Additional arguments
            
        Returns:
            List of normalized entropy rewards
        """
        if logprobs is None:
            # warning
            warnings.warn("logprobs is None, returning 0.0 rewards for all completions")
            return [0.0] * len(completions)

        # max ent RL for policy gradients is just the sum of the logprobs of the completion tokens
        # so we just return the sum of the logprobs
        if reduction == "sum":
            return torch.sum(-logprobs, dim=1).tolist()
        elif reduction == "mean":
            return torch.mean(-logprobs, dim=1).tolist()
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")

    return token_entropy_reward

def get_embedding_entropy_reward(
        embedding_entropy_reduction: str = "mean", # use pca?
        embedding_entropy_similarity: str = "cosine", # cosine, euclidean
        embedding_entropy_token: str = "last", # last, mean, concat
        embedding_entropy_hidden_state_reduction: tuple[int] = (-1, 100), # concat layer idxs in this range
    ):
    """
    Get a reward function that promotes diverse embeddings across generations by penalizing high cosine similarity.
    
    Args:
        embedding_entropy_reduction: The reduction method to use for the entropy reward.
        embedding_entropy_similarity: The similarity metric to use for the entropy reward.
        embedding_entropy_token: The token to use for the entropy reward.
        embedding_entropy_hidden_state_reduction: The hidden state reduction method to use for the entropy reward.
        max_similarity: Maximum allowed cosine similarity between embeddings (0.0 to 1.0)
    """
    def embedding_entropy_reward(completions, hidden_states=None, num_generations=None, **kwargs) -> list[float]:
        """Calculate reward based on embedding similarity between generations.
        
        Args:
            completions: List of model completions
            hidden_states: Hidden states (num_layers, B, L, H)
            num_generations: Number of generations per prompt
            **kwargs: Additional arguments
            
        Returns:
            List of rewards where higher values indicate more diverse embeddings
        """
        if hidden_states is None:
            warnings.warn("hidden_states is None, returning 0.0 rewards for all completions")
            return [0.0] * len(completions)
            
        if num_generations is None or num_generations == 1:
            warnings.warn("num_generations is None or 1, returning 0.0 rewards for all completions")
            return [0.0] * len(completions)
        
        # get the correct layer, concat layers in this range (B, L, H)
        embeddings = hidden_states.permute(1,2,0,3)[:,:,embedding_entropy_hidden_state_reduction[0]:embedding_entropy_hidden_state_reduction[1],:]
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))    # (B, L, H*num_layers)
        # get the correct token (B, H)
        if embedding_entropy_token == "last":
            # TODO using -2 to exclude the eos token, correct?
            embeddings = embeddings[:, -2, :]
        elif embedding_entropy_token == "mean":
            embeddings = embeddings.mean(dim=1)  # (B, H*num_selected_layers)
        elif embedding_entropy_token == "concat":
            embeddings = embeddings.reshape(embeddings.size(0), -1)  # (B, L*H*num_selected_layers)
        else:
            raise ValueError(f"Invalid token: {embedding_entropy_token}")
        
        # Reshape to group by prompt (num_generations per prompt)
        embeddings = embeddings.view(-1, num_generations, embeddings.size(-1))  # (B/G, G, H)
        # Normalize embeddings
        embeddings_norm = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)     
        
        if embedding_entropy_similarity == "cosine":
            # Compute cosine similarity between all pairs of embeddings within each group
            similarity = torch.matmul(embeddings_norm, embeddings_norm.transpose(-2, -1))  # (B/G, G, G)       
            # Mask out self-similarity and lower triangle
            similarity = similarity.triu(diagonal=1)
        elif embedding_entropy_similarity == "euclidean":
            # Compute pairwise Euclidean distances between embeddings
            # (a-b)^2 = a^2 + b^2 - 2ab
            # For normalized vectors, a^2 = b^2 = 1, so (a-b)^2 = 2 - 2ab
            # Therefore, distance = sqrt(2 - 2ab)
            dot_product = torch.matmul(embeddings_norm, embeddings_norm.transpose(-2, -1))  # (B/G, G, G)
            distance = torch.sqrt(2 - 2 * dot_product)  # (B/G, G, G)
            # Mask out self-distances and lower triangle
            distance = distance.triu(diagonal=1)
            similarity = -distance
        else:
            raise ValueError(f"Invalid similarity metric: {embedding_entropy_similarity}")
            
        # we want to maximize diversity, so we reward negative similarity
        rewards = -similarity  
        
        if embedding_entropy_reduction == "max":
            # max over last two dimensions
            rewards = rewards.max(dim=-1).max(dim=-1)[0]  # (B/G,)
        elif embedding_entropy_reduction == "mean":
            # Compute mean over all pairs
            num_pairs = (num_generations * (num_generations - 1)) / 2
            rewards = rewards.sum(dim=-1).sum(dim=-1) / num_pairs  # (B/G,)
        elif embedding_entropy_reduction == "sum":
            rewards = rewards.sum(dim=-1).sum(dim=-1)  # (B/G,)
        else:
            raise ValueError(f"Invalid reduction method: {embedding_entropy_reduction}")

        # repeat rewards for each generation
        rewards = rewards.repeat_interleave(num_generations)
        return rewards.tolist()

    return embedding_entropy_reward