# --- 1. Installation and Imports ---
# Install necessary libraries in the Colab environment

import os
import json
import random
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

#from google.colab import userdata # For securely accessing API keys

# --- Helper Functions ---

def test_model_connection(target_model):
    """Test if the model is accessible and working via the Google AI API."""
    test_prompt = "Say hello"
    test_input = "Hello, world!"
    try:
        response = run_google_rollout(target_model, test_prompt, test_input)
        return True, response
    except Exception as e:
        return False, str(e)

def log_message(message, type='info'):
    """Helper to format log messages with a timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    if type == 'success':
        return f"[{timestamp}] ✅ SUCCESS: {message}"
    if type == 'fail':
        return f"[{timestamp}] ❌ FAIL: {message}"
    if type == 'best':
        return f"[{timestamp}] ⭐ BEST: {message}"
    return f"[{timestamp}] ℹ️ INFO: {message}"

# ------ Ollama Implementation ------

def run_ollama_rollout(target_model, system_prompt, question, max_tokens=600):
    """
    Calls Ollama for the target model.
    This function performs a "rollout" for a given prompt and input.
    """
    url = "http://localhost:11434/api/chat"
    messages = [{"content": system_prompt, "role":"system"}, {"content": json.dumps(question)[1:-1].replace("```", ""), "role":"user"}]
    data = {
        "model": target_model,
        "messages": messages,
        "temperature": 1.2,
        "top_p": 0.95,
         "stream": False,
        "options": {
             "num_predict": max_tokens,
         }
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()["message"]["content"]


import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_mini_batch(dataset, batch_size=3):
    rng = np.random.default_rng()
    sampled_indices = rng.choice(len(dataset), size=batch_size, replace=False)

    # Get the sampled dataset
    mini_batch = dataset.select(sampled_indices)

    return sampled_indices, mini_batch

def get_batches(dataset, size_batch=3):
    """
    Splits a Hugging Face Dataset into batches.

    Args:
        dataset: Hugging Face Dataset object.
        size_batch: Number of examples per batch.

    Returns:
        List of batches, where each batch is a list of examples (dicts).
    """
    total = len(dataset)
    batches = []

    for i in range(0, total, size_batch):
        batch = dataset[i:i + size_batch]
        batches.append(batch)

    return batches

def mini_batch_rollout(target_model, prompt, mini_batch, eval_fun, max_threads=4):
    total_score = 0
    feedback_list = []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_info = {
        executor.submit(run_ollama_rollout, target_model, prompt, example['question']): (i, example)
        for i, example in enumerate(mini_batch)}

        results = [None] * len(future_to_info)
        feedback_list = [None] * len(future_to_info)
        total_score = 0

        for future in as_completed(future_to_info):
            i, example = future_to_info[future]
            try:
                completion = future.result()
                results[i] = completion
                eval_result = eval_fun(completion, example['answer'])
                total_score += eval_result['score']
                feedback_list[i] = f"==== Feedback for question {i} =====\n {eval_result['feedback']} \n================"
            except Exception as e:
                results[i] = None
                print(log_message(f'Failed to get a response from Ollama: {e}', type='Error'))
                print(f"Prompt used: {prompt}")
            
        return results, {"score": total_score/len(mini_batch), "feedback": feedback_list}



def batch_evaluation(target_model, prompt, dataset, score_row, eval_fun, max_threads=4):
    total_score = 0
    feedback_list = []
    completions = []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(run_ollama_rollout, target_model, prompt, example['question']) for example in dataset]
        for i, (example, future) in enumerate(zip(dataset, as_completed(futures))):
            completion = future.result()
            completions.append(completion)
            eval_result = eval_fun(completion, example['answer'])
            score_row[i] = eval_result['score']
            total_score += eval_result['score']
            feedback_list.append(eval_result['feedback'])

        return total_score

# --- Core GEPA Functions (Google API Implementation) ---

def run_google_rollout(target_model, prompt, input_text):
    """
    Calls the Google Generative AI API for the target model.
    This function performs a "rollout" for a given prompt and input.
    """
    full_prompt = f"{prompt}\n\nText: \"{input_text}\"\n\nResponse:"
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=600,
        temperature=0.7,
        top_p=0.95
    )
    try:
        response = target_model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        # Handle cases where the model response might be empty or blocked
        if not response.parts:
            raise Exception("Model returned an empty response. This could be due to safety filters.")
        return response.text
    except Exception as e:
        # Provide a more specific error message for API key issues
        if "api_key" in str(e).lower():
            raise Exception(f"Google AI API Error: Authorization failed. Ensure your Gemini API Key is correct and enabled.")
        raise Exception(f"Google AI API Error during rollout: {str(e)}")
