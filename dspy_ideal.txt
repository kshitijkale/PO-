# Tutorial: GEPA for AIME (Math)
In this tutorial, we optimize GPT-4.1 Mini's Chain of Thought (`dspy.ChainOfThought`) for solving math problems (AIME) using the `dspy.GEPA` optimizer!

<details>
<summary>Recommended: Set up MLflow Autologging to understand what's happening under the hood.</summary>

### MLflow DSPy Integration

<a href="https://mlflow.org/">MLflow</a> is an LLMOps tool that natively integrates with DSPy and offer explainability and experiment tracking. MLflow's autologging capability automatically tracks progress of GEPA optimization, as well as visualizes prompts and module executions as traces to understand the DSPy's behavior better. You can set up MLflow easily by following the four steps below.

**Visualize module executions as traces**

![MLflow Trace](./mlflow-tracing-gepa-aime.png)

**Automatically track optimization progress and results**

![MLflow Tracking](./mlflow-tracking-gepa-aime-optimization.png)


**Setup MLflow**

1. Install MLflow

```bash
%pip install mlflow>=3.0.0
```

2. Start MLflow UI in a separate terminal
```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

3. Connect the notebook to MLflow
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. Enabling autologging.

```python
mlflow.dspy.autolog(
    # Log the optimization progress
    log_compiles=True,
    # Log the evaluation results
    log_evals=True,
    # Log traces from module executions
    log_traces=True
)
```


To learn more about the integration, visit [MLflow DSPy Documentation](https://mlflow.org/docs/latest/llms/dspy/index.html) as well.
</details>

```python
api_key = input("Enter your OpenAI API key: ")
import dspy
lm = dspy.LM("openai/gpt-4.1-mini", temperature=1, api_key=api_key, max_tokens=32000)
dspy.configure(lm=lm)
```

### Loading the AIME dataset

The AIME exam consists of 2 problem sets of size 15 for each year. For this tutorial, we will use AIME problem sets from previous years (2022-2024) for optimization (amounting to total 3 years x 2 sets x 15 problems = 90 problems, split equally between train and validation sets), and test the performance on AIME 2025 (2 sets x 15 problems = 30 problems). Since AIME 2025 is a small set, we repeat it 5 times for statistical stability in evaluation.

```python
import dspy
from datasets import load_dataset

def init_dataset():
    train_split = load_dataset("AI-MO/aimo-validation-aime")['train']
    train_split = [
        dspy.Example({
            "problem": x['problem'],
            'solution': x['solution'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in train_split
    ]
    import random
    random.Random(0).shuffle(train_split)
    tot_num = len(train_split)

    test_split = load_dataset("MathArena/aime_2025")['train']
    test_split = [
        dspy.Example({
            "problem": x['problem'],
            'answer': x['answer'],
        }).with_inputs("problem")
        for x in test_split
    ]

    train_set = train_split[:int(0.5 * tot_num)]
    val_set = train_split[int(0.5 * tot_num):]
    test_set = test_split * 5

    return train_set, val_set, test_set
```

```python
train_set, val_set, test_set = init_dataset()

len(train_set), len(val_set), len(test_set)
```

Let's view an example task input

```python
print("Problem:")
print(train_set[0]['problem'])
print("\n\nSolution:")
print(train_set[0]['solution'])
print("\n\nAnswer:")
print(train_set[0]['answer'])
```

### Let's define the program: A simple `dspy.ChainOfThought`

```python
class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""
    problem = dspy.InputField()
    answer = dspy.OutputField()

program = dspy.ChainOfThought(GenerateResponse)
```

### Defining the evaluation metric
We simply check exact match between the predicted answer and the correct answer.

```python
def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        return 0
    return int(correct_answer == llm_answer)
```

### Evaluating unoptimized Chain Of Thought

```python
import dspy
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=32,
    display_table=True,
    display_progress=True
)

evaluate(program)
```

### Optimize the program with `dspy.GEPA`

GEPA is a _reflective_ prompt optimizer, and it's strength lies in being able to leverage additional sources of information, like the DSPy program's execution and evaluation pipelines, which provides GEPA more visibility into why the system got the score that it did, and then GEPA can introspect to identify how to improve the score. GEPA can also leverage additional supervision provided in this manner. For example, during optimization, we can return the correct solution's to the problems the program failed to solve.

We note that while such explicit supervision is not available in all scenarios, GEPA can work very flexibly with different forms of feedback (for example, using LLM-as-a-judge feedback shown in the PAPILLON tutorial, or just using answer labels, as shown in the facility-support tutorial).

Let's quickly modify the evaluation metric to become an optimization metric for GEPA, that can provide this additional supervision!

```python
def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    correct_answer = int(example['answer'])
    written_solution = example.get('solution', '')
    try:
        llm_answer = int(prediction.answer)
    except ValueError as e:
        feedback_text = f"The final answer must be a valid integer and nothing else. You responded with '{prediction.answer}', which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting."
        feedback_text += f" The correct answer is '{correct_answer}'."
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems and ensure your final answer is a valid integer."
        return dspy.Prediction(score=0, feedback=feedback_text)

    score = int(correct_answer == llm_answer)

    feedback_text = ""
    if score == 1:
        feedback_text = f"Your answer is correct. The correct answer is '{correct_answer}'."
    else:
        feedback_text = f"Your answer is incorrect. The correct answer is '{correct_answer}'."
    
    if written_solution:
        feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems."

    return dspy.Prediction(score=score, feedback=feedback_text)
```

```python
from dspy import GEPA

optimizer = GEPA(
    metric=metric_with_feedback,
    auto="light",
    num_threads=32,
    track_stats=True,
    reflection_minibatch_size=3,
    reflection_lm=dspy.LM(model="gpt-5", temperature=1.0, max_tokens=32000, api_key=api_key)
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)
```

### Let's see the prompt generated

```python
print(optimized_program.predict.signature.instructions)
```

It can be seen that what GEPA is doing here, is precomputing some reasoning to come up with a good plan for future task instances. Due to the improved performance in unseen validation set, we expect this prompt to generalize!

### Evaluating the Chain Of Thought optimized with GEPA

```python
evaluate(optimized_program)
```

GEPA was able to optimize the GPT-4.1 Mini's performance on AIME 2025 **from 46.6% score to 56.6%**, a 10% improvement, with just a budget of `auto="light"`!