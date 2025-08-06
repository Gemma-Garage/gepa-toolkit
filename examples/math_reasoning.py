######### Credits to Unsloth AI for most of the code in this file



reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


import re

def extract_number(text):
  match_seq = re.search(r"(?<=<SOLUTION>)[\d.,%$€+-]+(?=</SOLUTION>)", text)
  if match_seq:
    match_text = match_seq.group()
    match_text = match_text.replace(',', '.').replace('%', '').replace('$', '')
    try:
      return float(match_text)
    except:
      return None
  return None

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags = re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    scores = []
    feedback = ""
    for response in completions:
        score = 0
        # Match if format is seen exactly!
        if match_format.search(response) is not None:
          score += 3.0
          feedback = "Formatting correct"
        else:
          score -= 3.0
          feedback = "Formatting missing"
        scores.append(score)
    return scores, feedback

def match_format_approximately(completions, **kwargs):
    scores = []
    feedback = []
    for response in completions:
        score = 0
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        if response.count(reasoning_start) == 1:
          score += 0.5
          feedback.append(f"{reasoning_start} present. ")
        else:
          score -= 0.5
          feedback.append(f"{reasoning_start} missing. ")

        if response.count(reasoning_end)   == 1:
          score += 0.5
          feedback.append(f"{reasoning_end} present. ")
        else:
          score -= 0.5
          feedback.append(f"{reasoning_end} missing. ")

        if response.count(solution_start)  == 1:
          score += 0.5
          feedback.append(f"{solution_start} present. ")
        else:
          score -= 0.5
          feedback.append(f"{solution_start} missing. ")

        if response.count(solution_end)    == 1:
          score += 0.5
          feedback.append(f"{solution_end} present. ")
        else:
          score -= 0.5
          feedback.append(f"{solution_end} missing. ")

        scores.append(score)
    return scores, "".join(feedback)

def check_answer(completions, answer, **kwargs):
    responses = [completion for completion in completions]
    feedback = ""
    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        # Correct answer gets 3 points!
        if guess == true_answer:
            score += 3.0
            feedback = " Answer is correct! "
        # Match if spaces are seen
        elif guess.strip() == true_answer.strip():
            score += 1.5
            feedback = "Answer is correct but there are trailing spaces. "
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if   ratio >= 0.9 and ratio <= 1.1: score += 0.5
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.25
                else: score -= 1.0 # Penalize wrong answers
            except:
                score -= 0.5 # Penalize
            feedback = "Wrong answer! "
        scores.append(score)
    return scores, feedback

def check_numbers(completions, answers, **kwargs):
    responses = [completion for completion in completions]
    scores = []
    feedbacks = []
    for response, answer in zip(responses, answers):
      if extract_number(response) == float(answer):
        scores.append(1)
        feedbacks.append("Right answer!")
      else:
        scores.append(0)
        feedbacks.append("Wrong answer!")
        
    return scores, "".join(feedbacks)


def evaluation_and_feedback_function(output, answer, *kargs):
  try:
    score_format_exactly, feedback_format_exactly = match_format_exactly([output])
    score_format_approximately, feedback_format_approximately = match_format_approximately([output])
    score_check_answer, feedback_check_answer = check_answer([output], [answer])
    score_check_numbers, feedback_check_numbers = check_numbers([output], [answer])
    #print(f"score_format_exactly: {score_format_exactly}")
    #print(f"score_format_approximately: {score_format_approximately}")
    #print(f"score_check_numbers: {score_check_numbers}")
    return {
      "score": score_format_exactly[0] +
      score_format_approximately[0] +
      score_check_answer[0] +
      score_check_numbers[0],
      "feedback": ", ".join([feedback_format_exactly,
                            feedback_format_approximately,
                            feedback_check_answer,
                            feedback_check_numbers])
    }

  except:
    return {"score": 0, "feedback": ""}

