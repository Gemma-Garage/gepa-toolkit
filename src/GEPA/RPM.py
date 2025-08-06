#######################################################
# RPM - Reflective Prompt Mutation                    #
#######################################################

import time
import random
import os
from typing import List
from litellm import completion
from pydantic import BaseModel, Field, confloat
from typing import List
import re
import json

class TrajectoryScore(BaseModel):
    feedback: str
    score: confloat(ge=-10.0, le=100.0)

class TrajectoryGradingOutput(BaseModel):
    feedbacks: List[TrajectoryScore]

def call_with_backoff(func, max_retries=5, *args, **kwargs):
    delay = 0.1  # initial delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):  # customize based on error
                print(f"Rate limit hit, backing off... attempt {attempt+1}")
                time.sleep(delay)
                delay *= 2  # exponential backoff
                delay += random.uniform(0, 0.01)  # jitter
            else:
                raise e
    raise RuntimeError("Exceeded maximum retries due to rate limiting.")


def get_trajectory_feedback(judge_model_name, rubric, examples, size_group) -> TrajectoryGradingOutput:
    judge_prompt = f"""
You are a grader evaluating an agent response against a goal rubric.
Give the trajectory a score between -10 and 100 and a quick feedback (less than 150 chars) on the trajectory's performance, if
the answer is numerically correct and if the formatting was completely or partially followed. If you have {size_group} trajectories 
you must output {size_group} feedbacks.

Rubric:
{rubric}

Trajectories:
{examples}
"""

    try:
        response = completion(
            model=judge_model_name,  # LiteLLM-style model name for Gemini
            messages=[{"role": "user", "content": judge_prompt}],
            response_format=TrajectoryGradingOutput,  # Enforce Pydantic format
            max_retries=2,  # Optional: retry if model returns malformed output,
            max_tokens=1024
        )
        return response  # this will be a parsed `TrajectoryGradingOutput` instance

    except Exception as e:
        print(f"Validation or generation error: {e}")
        raise


def reflect_and_propose_new_prompt(reflector_model, reflector_model_name, current_prompt, eval_result):
    """
    Performs the Reflective Prompt Mutation step using a powerful LLM (e.g., Gemini Pro).
    """
    # examples_text = '---'.join(
    #     f'Task Input: "{e["input"]}"\nGenerated Output: "{e["output"]}"\nFeedback:\n{e["feedback"]}\n\n'
    #     for e in examples
    # )
    examples = []
    for i, (question, output, answer) in enumerate(zip(eval_result['input'], eval_result['output'], eval_result['answers'])):
        examples.append(f"Question {i}: {question} \n Model response: {output} \n Predicted answer: {output} Expected answer: {answer} \n==============\n")
    
    response_feedbacks = get_trajectory_feedback(reflector_model_name, current_prompt, "".join(examples), len(examples))
    first_choice = response_feedbacks.choices[0]
    content = first_choice.message.content or "{}"
    feedbacks = TrajectoryGradingOutput.model_validate_json(content).feedbacks
    feedbacks = [element.feedback for element in feedbacks]
    #print(f"Examples: {"".join(examples)}")
    reflection_prompt = f"""I provided an assistant with the following instructions to perform a task for me:
‘‘‘
{current_prompt}
‘‘‘
The following are examples of different task inputs provided to the assistant
along with the assistant’s response for each of them, and some feedback on how
the assistant’s response could be better:
‘‘‘
{" New feedback: ".join(examples)}
‘‘‘
Feedback:
‘‘‘
{"".join(feedbacks)}
‘‘‘

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task
description about the task I wish to solve with the assistant.
Read all the assistant responses and the corresponding feedback. Identify all
niche and domain specific factual information about the task and include it in
the instruction, as a lot of it may not be available to the assistant in the
future. The assistant may have utilized a generalizable strategy to solve the
task, if so, include that in the instruction as well. Understand the root cause
for mistakes and reflect this in the prompt.
Return ONLY the new prompt in the response."""

    try:
        response = reflector_model.generate_content(reflection_prompt)
        if not response.parts:
             raise Exception("Reflector model returned an empty response. This could be due to safety filters.")
        raw_prompt = response.text.strip()
        #prompt = raw_prompt.replace('%', '%%').replace('"', '\\"')
        prompt = json.dumps(raw_prompt.replace('```', ''))[1:-1]
        return prompt
    except Exception as e:
        raise Exception(f"Gemini API Error during reflection: {str(e)}. Check your Gemini API Key.")


def select_candidate_for_mutation(candidate_pool, num_tasks):
    """Selects the next candidate to mutate based on the Pareto-based strategy."""
    if len(candidate_pool) == 1:
        return candidate_pool[0]

    best_scores_per_task = [-1.0] * num_tasks
    for candidate in candidate_pool:
        for i in range(num_tasks):
            if candidate["scores"][i] > best_scores_per_task[i]:
                best_scores_per_task[i] = candidate["scores"][i]

    pareto_front_ids = set()
    for i in range(num_tasks):
        for candidate in candidate_pool:
            if abs(candidate["scores"][i] - best_scores_per_task[i]) < 1e-6:
                pareto_front_ids.add(candidate["id"])

    if not pareto_front_ids:
        return max(candidate_pool, key=lambda c: c["avg_score"])

    selected_id = random.choice(list(pareto_front_ids))
    return next(c for c in candidate_pool if c["id"] == selected_id)
