# GEPA: Genetic Pareto Reflective Prompt Evolution Implementation

⚠️ **IMPORTANT**: This is an independent implementation and is not affiliated with the authors of the original paper. Any mistakes in the implementation are mine alone.

This repository contains an implementation of **GEPA (Genetic-Pareto)**, a prompt optimization technique that uses reflective prompt evolution to outperform reinforcement learning methods. This implementation is based on the research paper:

**"GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning"** by Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab.



📄 **Paper**: [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)

Code freely inspired by Richard Aragon's awesome implementation in the following [notebook](https://colab.research.google.com/drive/1FPrjP5yTzr1hdL_S_77MyhVi4JOMPTpy?usp=sharing)

## What is Genetic-Pareto (GEPA)?

GEPA is a novel prompt optimization approach that leverages the interpretable nature of language to learn high-level rules from trial and error. Unlike traditional reinforcement learning methods that rely on sparse, scalar rewards, GEPA uses natural language reflection to:

1. **Sample system-level trajectories** (reasoning, tool calls, outputs)
2. **Reflect on them in natural language** to diagnose problems
3. **Propose and test prompt updates** based on insights
4. **Combine complementary lessons** from the Pareto frontier of attempts

The key innovation is that GEPA can turn even just a few rollouts into significant quality gains, often outperforming methods like Group Relative Policy Optimization (GRPO) by 10-20% while using up to 35x fewer rollouts.

## Features

- ✅ **Ollama Integration**: Currently supports Ollama for model inference
- ✅ **Reflective Prompt Evolution**: Uses natural language reflection for prompt optimization
- ✅ **Pareto Frontier Optimization**: Combines complementary lessons from multiple attempts
- ✅ **Custom Reward Functions**: Define your own evaluation criteria
- ✅ **Batch Evaluation**: Efficient parallel evaluation of prompts
- ✅ **Mini-batch Rollouts**: Optimized sampling for reflection

## Installation

```bash
pip install -r requirements.txt
```

## Setup

### 1. Environment Variables

Set your API keys through environment variables:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### 2. Define Your Reward Function

Create a custom evaluation function that takes model outputs and expected answers, returning a score and feedback:

```python
def your_evaluation_function(output, answer, **kwargs):
    """
    Custom evaluation function for your task.
    
    Args:
        output: Model's response
        answer: Expected answer
        **kwargs: Additional arguments
    
    Returns:
        dict: Contains 'score' (float) and 'feedback' (str)
    """
    # Your evaluation logic here
    score = 0.0
    feedback = ""
    
    # Example: Check if answer is correct
    if output.strip() == answer.strip():
        score = 1.0
        feedback = "Correct answer!"
    else:
        score = 0.0
        feedback = "Incorrect answer."
    
    return {"score": score, "feedback": feedback}
```

## Usage

### Basic Example

```python
from GEPA.GEPA import GEPATrainer
from datasets import load_dataset
import os

# Set your API key
os.environ['GEMINI_API_KEY'] = "your_gemini_api_key_here"

# Load your dataset
dataset = load_dataset("your_dataset", split="train").select(range(10))

# Define your evaluation function
def my_eval_function(output, answer, **kwargs):
    # Your evaluation logic here
    return {"score": 1.0 if output == answer else 0.0, "feedback": "Evaluation complete"}

# Initialize GEPA trainer
gepa_trainer = GEPATrainer(
    target_model_name='gemma3:1b',      # Ollama model name
    reflector_model_name='gemini-2.0-flash',  # Reflection model
    train_dataset=dataset,
    eval_fun=my_eval_function,
    seed_prompt="Your initial prompt here",
    budget=500  # Number of rollouts
)

# Run optimization
gepa_trainer.train()

# Get the best prompt
best_prompt = gepa_trainer.get_best_prompt()
print(f"Best prompt: {best_prompt}")
```

### Math Reasoning Example

See `examples/math_reasoning.py` for a complete example with mathematical reasoning tasks, including:

- Structured output formatting
- Multi-criteria evaluation
- Number extraction and validation

## How It Works

### 1. Initialization
- Starts with a seed prompt
- Evaluates initial performance on training data

### 2. Reflective Evolution Loop
- **Selection**: Chooses a candidate prompt from the pool
- **Mini-batch Sampling**: Samples a small batch for reflection
- **Rollout**: Runs the current prompt on the mini-batch
- **Reflection**: Uses a reflection model to analyze performance
- **Mutation**: Generates an improved prompt based on insights
- **Evaluation**: Tests the new prompt on the full dataset
- **Selection**: Keeps the prompt if it improves performance

### 3. Pareto Frontier
- Maintains a pool of diverse, high-performing prompts
- Combines complementary strategies from different successful attempts

## Key Components

- **`GEPATrainer`**: Main optimization class
- **`RPM.py`**: Reflective Prompt Mutation logic
- **`rollout.py`**: Batch evaluation and mini-batch generation
- **`utils.py`**: Utility functions

## Configuration

### GEPATrainer Parameters

- `target_model_name`: Ollama model for task execution
- `reflector_model_name`: Model for reflection and prompt generation
- `train_dataset`: Training dataset
- `eval_fun`: Custom evaluation function
- `seed_prompt`: Initial prompt to optimize
- `budget`: Maximum number of rollouts
- `rollout_type`: Currently supports 'ollama'

## Examples

Check the `examples/` directory for:

- `example_1.py`: Basic usage example
- `math_reasoning.py`: Complex mathematical reasoning task
- `Experiments_with_GEPA_Reflective_Prompt_Evolution.ipynb`: Jupyter notebook with experiments

## Requirements

- Python 3.8+
- google-generativeai
- datasets
- litellm
- pydantic

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{agrawal2025gepa,
  title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal, Lakshya A and Tan, Shangyin and Soylu, Dilara and Ziems, Noah and Khare, Rishi and Opsahl-Ong, Krista and Singhvi, Arnav and Shandilya, Herumb and Ryan, Michael J and Jiang, Meng and Potts, Christopher and Sen, Koushik and Dimakis, Alexandros G and Stoica, Ion and Klein, Dan and Zaharia, Matei and Khattab, Omar},
  journal={arXiv preprint arXiv:2507.19457},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This implementation is provided for research purposes. Please refer to the original paper for licensing details.

## Acknowledgments

This implementation is based on the research by the authors of the GEPA paper. Special thanks to the research team for their groundbreaking work on reflective prompt evolution.

---

**Note**: This implementation currently supports Ollama inference. For other inference backends, modifications to the rollout functions may be required. 