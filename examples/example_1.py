from GEPA.GEPA import GEPATrainer
from datasets import load_dataset
from math_reasoning import evaluation_and_feedback_function
import os
import sys

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


GEMINI_API_KEY = #...your gemini api key here

os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY

dataset = load_dataset("openai/gsm8k", "main", split = "test").select(range(5))

gepa_trainer = GEPATrainer('gemma3:1b', 
                           'gemini-2.0-flash', 
                           dataset, 
                           evaluation_and_feedback_function, 
                           system_prompt)

gepa_trainer.train()
print(f"Best prompt after GEPA optmization: {gepa_trainer.get_best_prompt()}")




