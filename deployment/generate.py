import openai
import argparse
import torch
import os
import anthropic
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import get_prompt
from dotenv import load_dotenv

openai.api_key = "FILL IN YOUR KEY HERE."
# openai.api_base = "http://localhost:8000/v1"
enc = tiktoken.get_encoding("cl100k_base")

DEVELOPMENT_TASKS = ["feedback", "airline-reviews", "textual-entailment", "chatgpt-prompt", "ett-m2", "ili", "handwriting", "ethanol-concentration", "media-campaign-cost", "wild-blueberry-yield", "spaceship-titanic", "enzyme-substrate"]
DEPLOYMENT_TASKS = ['smoker-status', 'mohs-hardness', 'bitcoin-price-prediction', 'heartbeat', 'webmd-reviews', 'cirrhosis-outcomes', 'software-defects', 'hotel-reviews', 'electricity', 'detect-ai-generation', 'weather', 'self-regulation-scp1', 'uwave-gesture-library', 'traffic', 'boolq', 'crab-age', 'concrete-strength', 'jigsaw',
                    'airport_eta']

def get_args():
    parser = argparse.ArgumentParser()
    # Model Information
    parser.add_argument("--llm", default="deepseek-chat")		# LLM name # 原来是gpt-3.5-turbo-16k
    parser.add_argument("--task", default="airport_eta")   # ML Task name
    # Context Configuration
    parser.add_argument("--shot", default=1, type=int)              # Number of examples in context
    parser.add_argument("--retrieval", default=True,               # Whether activate retrieval
                        action='store_true')
    parser.add_argument("--raw", default=False,                     # Whether use raw cases
                        action='store_true')
    # Generation Configuration
    parser.add_argument("--temperature", default=0.7, type=float)   # Temperature (fixed)
    parser.add_argument("--trials", default=10, type=int)    		# Number of trials (fixed)

    args = parser.parse_args()
    return args


def configure_api(llm):
    if llm.startswith("deepseek"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        api_base = os.getenv("DEEPSEEK_BASE_URL")
        missing_key_name = "DEEPSEEK_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_BASE_URL")
        missing_key_name = "OPENAI_API_KEY"

    if not api_key:
        raise ValueError(f"Missing API key. Please set environment variable {missing_key_name}.")

    if hasattr(openai, "OpenAI"):
        return openai.OpenAI(api_key=api_key, base_url=api_base)

    openai.api_key = api_key
    setattr(openai, "api_base", api_base)
    return None


def extract_code(raw_completion):
    if "```python" in raw_completion:
        return raw_completion.split("```python", 1)[1].split("```", 1)[0]
    if "```" in raw_completion:
        return raw_completion.split("```", 1)[1].split("```", 1)[0]
    return raw_completion


def generation(prompt, llm, client=None, temperature=0.7, log_file=None):
    raw_request = {
        "model": llm,
        "temperature": temperature,
        "max_tokens": 4000,
        "stop": [] or None,
    }
    iteration = 0
    completion = None
    raw_completion = ""
    while iteration < 50:
        try:
            messages = [{"role": "user", "content": prompt}]
            if client is not None:
                response = client.chat.completions.create(messages=messages, **raw_request)
                raw_completion = response.choices[0].message.content
            else:
                response = getattr(openai, "ChatCompletion").create(**{"messages": messages, **raw_request})
                raw_completion = response["choices"][0]["message"]["content"]
            completion = extract_code(raw_completion)
            if not completion.strip(" \n"):
                continue
            break
        except Exception as e:
            iteration += 1
            print(f"===== Retry: {iteration} =====")
            print(f"Error occurs when calling API: {e}")
        continue
    if not completion:
        completion = ""
    print(completion)
    log_to_file(log_file, prompt, raw_completion)
    return completion
    

def log_to_file(log_file, prompt, completion):
    """ Log the prompt and completion to a file."""
    num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
    num_sample_tokens = len(enc.encode(completion))
    
    # Logging for finetuning
    with open(log_file, "wt") as f:
        f.write(prompt)
        f.write("\n[This is a split string for finetuning]\n")
        f.write(completion)
        f.write("\n[This is a split string for counting tokens]\n")
        f.write(f"Prompt: {num_prompt_tokens}, Completion: {num_sample_tokens}")


if __name__ == '__main__':
    load_dotenv("../.env")
    args = get_args()

    client = configure_api(args.llm)
        
    # Load Tasks
    if args.task == "all":
        tasks_to_solve = DEPLOYMENT_TASKS
    else:
        assert args.task in DEPLOYMENT_TASKS
        tasks_to_solve = [args.task]
    
    # Pathname
    prefix = f"{args.llm}_{args.retrieval}_{args.shot}" if not args.raw else f"{args.llm}_{args.retrieval}_{args.shot}_raw"
    
    # Create the path for generation results
    pathname = f"./codes/{prefix}"
    if not os.path.exists(pathname):
        os.makedirs(pathname)
       
    # Create Finetune Logs
    finetune_dir = f"./codes/{prefix}/finetune_log"
    if not os.path.exists(finetune_dir):
        os.makedirs(finetune_dir)
    
    for task in tasks_to_solve:
        print(f"Processing Task: {task}")
        tmp_pathname = f"{pathname}/{task}"
        if not os.path.exists(tmp_pathname):
            os.makedirs(tmp_pathname)
        temp_finetunedir = f"{finetune_dir}/{task}"
        if not os.path.exists(temp_finetunedir):
            os.makedirs(temp_finetunedir)
        for idx in range(args.trials):
            prompt = get_prompt(task, context_num=args.shot, strategy="retrieval" if args.retrieval else "random", raw=args.raw)
            response = generation(prompt, args.llm, client=client, temperature=args.temperature, log_file=f"{temp_finetunedir}/{idx}.txt")
            filename = f"{tmp_pathname}/train_{idx}.py"
            with open(filename, "wt") as file:
                file.write(response)
            
        