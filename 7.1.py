import os
from ldata import LetterConcatenation, EvaluationMetric
from ldata.utils import NumberListOperation
from lmodels import DeepSeekModel
from lmethods import MetaPrompting, RecursivePrompting

os.environ["OPENROUTER_API_KEY"] = "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"
print(os.getenv("OPENROUTER_API_KEY"))

# Проверка CSV (для отладки)
import pandas as pd
df = pd.read_csv("letter_concat.csv")
print("Columns in CSV:", df.columns.tolist())
print("Sample data:\n", df.head())

# Настройка бенчмарка
benchmark = LetterConcatenation(
    LetterConcatenation.Config(
        data_path="letter_concat.csv",
        letter_idx=1
    )
)
print("Full set inputs:", benchmark.full_set.inputs)
print("Full set targets:", benchmark.full_set.targets)
print("Test set length:", benchmark.test_len)

# Настройка модели
model = DeepSeekModel(config={})

# Настройка методов
cot = MetaPrompting(
    model,
    MetaPrompting.Config(prompt_path="prompts/cot.md")
)
rd = RecursivePrompting(
    model,
    RecursivePrompting.Config(
        unit_prompt_path="prompts/cot.md",
        split_prompt_path="prompts/decompose.md",
        merge_prompt_path="prompts/merge.md",
        max_nodes=20
    )
)

# Запуск эксперимента
for method in [cot, rd]:
    inputs, targets, outputs, solutions, scores, agg_score, info = benchmark.evaluate_subject(
        lambda inputs: method.generate(inputs, max_tokens=2000),
        n_samples=min(benchmark.test_len, 2),
        metric=EvaluationMetric.EXACT,
        aggregation_fn=NumberListOperation.MEAN
    )
    print(f"Method {method.name}:")
    print(f"- Inputs: {inputs}")
    print(f"- Targets: {targets}")
    print(f"- Raw outputs: {outputs}")
    print(f"- Parsed solutions: {solutions}")
    print(f"- All scores: {scores}")
    print(f"- Aggregated score (mean): {agg_score}")
    if info is not None:
        cost = 0.05 * info.usage.context_tokens + 0.15 * info.usage.output_tokens
        print(f"- Cost: {cost}")
    else:
        print("- Cost: Not available (info is None)")