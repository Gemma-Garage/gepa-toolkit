#######################################################
# Main GEPA Class                                     #
#######################################################



import json
import os
import google.generativeai as genai
from .utils import log_message
from .RPM import select_candidate_for_mutation, reflect_and_propose_new_prompt
from .rollout import mini_batch_rollout, batch_evaluation, generate_mini_batch




class GEPATrainer():

    def __init__(self, 
                target_model_name, 
                reflector_model_name, 
                train_dataset, 
                eval_fun, 
                seed_prompt,
                budget=500,
                rollout_type='ollama'):

        self.target_model_name = target_model_name
        self.reflector_model_name = reflector_model_name
        self.train_dataset = train_dataset
        self.rollout_fun, self.batch_evaluation_fun = self.define_rollout(rollout_type)
        self.eval_fun = eval_fun
        self.seed_prompt = seed_prompt
        self.api_key = os.environ['GEMINI_API_KEY']
        self.budget = budget
        self.best_prompt = seed_prompt

    def define_rollout(self, rollout_type):
        if rollout_type == "ollama":
            return mini_batch_rollout, batch_evaluation

    def get_best_prompt(self):
        return self.best_prompt()

    def train(self):
        self.run_gepa_optimization(self.target_model_name, 
                              self.reflector_model_name,
                              self.seed_prompt,
                              self.train_dataset,
                              500)

    def run_gepa_optimization(self, target_model_name, reflector_model_name, seed_prompt, training_data, budget):
        """
        The main function that orchestrates the GEPA optimization process.
        """
        # --- Initialization ---
        print(log_message("Starting GEPA Optimization Process..."))
        GEMINI_API_KEY = self.api_key
        genai.configure(api_key=GEMINI_API_KEY)
        target_model = target_model_name #genai.GenerativeModel(target_model_name)
        reflector_model = genai.GenerativeModel(reflector_model_name)

        rollout_count = 0
        candidate_pool = []
        best_candidate = {"prompt": "Initializing...", "avg_score": -1.0}

        # --- Initial Evaluation of Seed Prompt ---
        print("\n" + "="*50)
        print(log_message("Phase 1: Evaluating Initial Seed Prompt"))
        initial_candidate = {"id": 0, "prompt": seed_prompt, "parentId": None, "scores": [0.0] * len(training_data), "avg_score": 0.0}
        total_score = 0.0
        total_score = batch_evaluation(target_model, seed_prompt, training_data, initial_candidate['scores'], self.eval_fun, max_threads=4)
        rollout_count += 1
        initial_candidate["avg_score"] = total_score / len(training_data) if training_data else 0.0
        candidate_pool.append(initial_candidate)
        best_candidate = initial_candidate

        print(log_message(f"Seed prompt initial score: {initial_candidate['avg_score']:.2f}", 'best'))
        print(f"Current Best Prompt:\n---\n{best_candidate['prompt']}\n---")


        # --- Main Optimization Loop ---
        print("\n" + "="*50)
        print(log_message(f"Phase 2: Starting Optimization Loop (Budget: {budget} rollouts)"))
        while rollout_count < budget:
            iteration_start_rollouts = rollout_count
            print(log_message(f"--- Iteration Start (Rollouts: {rollout_count}/{budget}) ---"))

            parent_candidate = select_candidate_for_mutation(candidate_pool, len(training_data))
            print(log_message(f"Selected candidate #{parent_candidate['id']} (Score: {parent_candidate['avg_score']:.2f}) for mutation."))

            #task_index = random.randint(0, len(training_data) - 1)
            #reflection_task = training_data[task_index]
            indexes, mini_batch = generate_mini_batch(training_data, batch_size=3)
            questions = [question for question in mini_batch['question']]
            answers = [answer for answer in mini_batch['answer']]
            print(log_message(f"Performing reflective mutation using indices {indexes}..."))

            try:
                rollout_count += 1
                rollouts, eval_result = mini_batch_rollout(target_model, parent_candidate["prompt"], mini_batch, self.eval_fun)

                new_prompt = reflect_and_propose_new_prompt(reflector_model, reflector_model_name, parent_candidate["prompt"], {
                    "input": questions, "answers": answers, "output": rollouts, "feedback": eval_result["feedback"]
                })

                new_candidate = {"id": len(candidate_pool), "prompt": new_prompt, "parentId": parent_candidate["id"], "scores": [0.0] * len(training_data), "avg_score": 0.0}
                print(log_message(f"Generated new candidate prompt #{new_candidate['id']}."))
                
                print(f"New candidate: {new_candidate["prompt"]}")
                _, new_prompt_eval_result = mini_batch_rollout(target_model, new_candidate["prompt"], mini_batch, self.eval_fun)

                #if performance in mini batch is not improved discard mutated prompt
                if new_prompt_eval_result['score'] < eval_result['score']:
                    print(log_message("Performance in mini batch didn't improve, prompt discarted"))
                    continue

                new_total_score = 0.0
                
                new_total_score = batch_evaluation(target_model, new_candidate['prompt'], training_data, new_candidate['scores'], self.eval_fun, max_threads=4)
                print(log_message(f"-> update scores for new_candidate: {new_candidate['scores']} "))
                new_candidate["avg_score"] = new_total_score / len(training_data) if training_data else 0.0

                if new_candidate["avg_score"] > parent_candidate["avg_score"]:
                    print(log_message(f"New candidate #{new_candidate['id']} improved! Score: {new_candidate['avg_score']:.2f} > {parent_candidate['avg_score']:.2f}", 'success'))
                    candidate_pool.append(new_candidate)
                    if new_candidate["avg_score"] > best_candidate["avg_score"]:
                        best_candidate = new_candidate
                        print(log_message("NEW BEST PROMPT FOUND!", 'best'))
                        print(f"Current Best Prompt:\n---\n{best_candidate['prompt']}\n---")
                        self.best_prompt = best_candidate['prompt']
                else:
                    print(log_message(f"New candidate #{new_candidate['id']} did not improve. Score: {new_candidate['avg_score']:.2f}. Discarding.", 'fail'))

            except Exception as e:
                print(log_message(f"Error in optimization iteration: {str(e)}", 'fail'))
                # Ensure rollout is counted even if a step fails before the evaluation loop
                if iteration_start_rollouts == rollout_count:
                    rollout_count += 1

        print("\n" + "="*50)
        print(log_message("Optimization budget exhausted. Finished.", 'best'))
        print(f"Final Best Prompt (Score: {best_candidate['avg_score']:.2f}):")
        print(f"\n{best_candidate['prompt']}\n")
        print("="*50)
        return best_candidate
