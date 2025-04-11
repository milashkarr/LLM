import os
import re
from ldata import LetterConcatenation, EvaluationMetric
from ldata.utils import NumberListOperation
from lmethods.utils import Usage
from lmodels import DeepSeekModel
from lmethods import MetaPrompting, RecursivePrompting, Method
from openai import OpenAI
import numpy as np

os.environ["OPENROUTER_API_KEY"] = "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"
print(os.getenv("OPENROUTER_API_KEY"))

# # Проверка CSV (для отладки)
# import pandas as pd
# df = pd.read_csv("letter_concat.csv")
# print("Columns in CSV:", df.columns.tolist())
# print("Sample data:\n", df.head())

# Настройка бенчмарка
benchmark = LetterConcatenation(
    LetterConcatenation.Config(
        data_path="letter_concat.csv",
        letter_idx=1
    )
)
# print("Full set inputs:", benchmark.full_set.inputs)
# print("Full set targets:", benchmark.full_set.targets)
# print("Test set length:", benchmark.test_len)

# Настройка модели с отладкой
class DebugDeepSeekModel(DeepSeekModel):
    def __init__(self, config):
        self._config = config
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openrouter_api_key,
            max_retries=0,
            timeout=20
        )
        # print(f"API key used: {self.openrouter_api_key}")

    def generate(self, prompts, max_tokens=500):
        # print(f"DebugDeepSeekModel generate - Prompts: {prompts}")
        try:
            response = self.generate_response(
                messages=[{"role": "user", "content": prompts[0]}],
                response_format=None
            )
            # print(f"DebugDeepSeekModel generate - Raw API response: {response}")
            try:
                match = re.search(r'```(.*?)```', response, re.DOTALL) or \
                        re.search(r'Final answer: \[?(.*?)\]?', response) or \
                        re.search(r'\\boxed{(.*?)}', response) or \
                        re.search(r'\"(.*?)\"$', response.strip()) or \
                        re.search(r'result is:?\s*\"?(.*?)\"?$', response)
                parsed_response = match.group(1).strip() if match else response.strip()
                parsed_response = re.sub(r'[\[\]\'"]', '', parsed_response).strip()
            except Exception as parse_error:
                # print(f"DebugDeepSeekModel generate - Parsing error: {parse_error}")
                parsed_response = response.strip()
            output = [parsed_response]
            info = Method.GenerationInfo(usage=Usage(n_calls=1, n_tokens_context=0, n_tokens_output=0))
            # print(f"DebugDeepSeekModel generate - Parsed output: {output}")
            # print(f"DebugDeepSeekModel generate - Info: {info}")
            return output, info
        except Exception as e:
            # print(f"DebugDeepSeekModel generate - Error: {type(e).__name__}: {str(e)}")
            return [""], Method.GenerationInfo(usage=Usage())

    def generate_response(self, messages, response_format=None):
        # print(f"DebugDeepSeekModel generate_response - Messages: {messages}")
        try:
            api_response = self._client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=messages,
                max_tokens=self._config.get("max_response_tokens", 4095),
                temperature=self._config.get("temperature", 0.0),
                extra_headers={
                    "HTTP-Referer": self._config.get("http_referer", "https://your-site.com"),
                    "X-Title": self._config.get("app_name", "Debug Application")
                }
            )
            # print(f"DebugDeepSeekModel generate_response - Full API response: {api_response}")
            content = api_response.choices[0].message.content
            return content
        except Exception as e:
            # print(f"DebugDeepSeekModel generate_response - Error: {type(e).__name__}: {str(e)}")
            raise

model = DebugDeepSeekModel(config={})
# print(f"Model class: {model.__class__.__name__}")

# # Тестовый вызов для проверки
# test_prompt = ["Test prompt: Hello"]
# test_output, test_info = model.generate(test_prompt)
# print(f"Test output: {test_output}")
# print(f"Test info: {test_info}")

# Настройка методов с отладкой
class DebugMetaPrompting(MetaPrompting):
    def __init__(self, model, config):
        super().__init__(model, config)
        # with open(config.prompt_path, 'r') as f:
        #     print(f"DebugMetaPrompting prompt content: {f.read()}")

    def generate(self, context, shots=MetaPrompting.ShotsCollection(), max_tokens=500):
        # print(f"DebugMetaPrompting generate - Context: {context}")
        # print(f"Model class in MetaPrompting: {self._model.__class__.__name__}")
        if isinstance(context, (list, tuple, np.ndarray)):
            results = []
            infos = []
            for ctx in context:
                result, info = self._generate_impl(ctx, shots, max_tokens)
                results.append(result)
                infos.append(info)
            combined_info = infos[0]
            for info in infos[1:]:
                combined_info += info
            return results, combined_info
        else:
            return self._generate_impl(context, shots, max_tokens)

    def _generate_impl(self, context, shots, max_tokens):
        # print(f"DebugMetaPrompting _generate_impl - Context: {context}")
        # print(f"DebugMetaPrompting _generate_impl - Shots: {shots}")
        prompt = self._prompt.format(problem=str(context), shots="")
        # print(f"DebugMetaPrompting _generate_impl - Formatted prompt: {prompt}")
        output, info = self._model.generate([prompt], max_tokens=max_tokens)
        # print(f"DebugMetaPrompting _generate_impl - Model output: {output}")
        # print(f"DebugMetaPrompting _generate_impl - Model info: {info}")
        return output[0], self.generation_info_cls(usage=info.usage)

class DebugRecursivePrompting(RecursivePrompting):
    def __init__(self, model, config):
        super().__init__(model, config)
        # with open(config.unit_prompt_path, 'r') as f:
        #     print(f"DebugRecursivePrompting unit prompt content: {f.read()}")

    def generate(self, context, max_tokens=2000):
        # print(f"DebugRecursivePrompting generate - Context: {context}")
        # print(f"Model class in RecursivePrompting: {self._model.__class__.__name__}")
        prompt = str(context[0]) if isinstance(context, (list, tuple)) and len(context) > 0 else str(context)
        # print(f"DebugRecursivePrompting generate - Processed prompt: {prompt}")
        output, info = self._model.generate([prompt], max_tokens=max_tokens)
        # print(f"DebugRecursivePrompting generate - Model output: {output}")
        # print(f"DebugRecursivePrompting generate - Model info: {info}")
        return output, self.generation_info_cls(usage=info.usage if info else Usage())

cot = DebugMetaPrompting(model, MetaPrompting.Config(prompt_path="prompts/cot.md"))
rd = DebugRecursivePrompting(
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
    # print(f"- Inputs: {inputs}")
    # print(f"- Targets: {targets}")
    # print(f"- Raw outputs: {outputs}")
    print(f"- Parsed solutions: {solutions}")
    print(f"- All scores: {scores}")
    print(f"- Aggregated score (mean): {agg_score}")
    if info is not None:
        # print(f"- Info: {info}")
        cost = 0.05 * info.usage.n_tokens_context + 0.15 * info.usage.n_tokens_output
        print(f"- Cost: {cost}")
    # else:
    #     print("- Cost: Not available (info is None)")
