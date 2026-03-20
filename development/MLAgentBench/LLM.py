""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
import tiktoken
from .schema import TooLongPromptError, LLMError
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True), override=False)

enc = tiktoken.get_encoding("cl100k_base")
STATISTICAL_DIR = None
FINETUNE_STEP = 0
    
try:
    import openai
except Exception as e:
    openai = None
    print(e)
    print("Could not import openai package.")

# 用于日志的通用前后缀，避免依赖 anthropic 常量。
HUMAN_PROMPT = "Human:"
AI_PROMPT = "Assistant:"


def _get_api_credentials(model):
    # 仅保留 OpenAI-compatible 的统一凭证入口：
    # - deepseek* 模型读取 DEEPSEEK_* 环境变量（默认）
    # - 其他模型可读取 OPENAI_* 环境变量（兼容保留）
    if model.startswith("deepseek"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        api_base = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        missing_key_name = "DEEPSEEK_API_KEY"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_BASE_URL")
        missing_key_name = "OPENAI_API_KEY"

    if not api_key:
        raise LLMError(f"Missing API key. Please set environment variable {missing_key_name}.")

    return api_key, api_base


def _get_openai_client(model):
    # 兼容 openai 新旧 SDK：
    # - 新版：openai.OpenAI(...)
    # - 旧版：设置 openai.api_key / openai.api_base
    if openai is None:
        raise LLMError("openai package is not installed")

    api_key, api_base = _get_api_credentials(model)
    if hasattr(openai, "OpenAI"):
        return openai.OpenAI(api_key=api_key, base_url=api_base)

    openai.api_key = api_key
    if api_base:
        setattr(openai, "api_base", api_base)
    return None


def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    # 通用日志函数：当前由 OpenAI-compatible 通道复用。
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")
    
    # Logging for finetuning
    if not STATISTICAL_DIR:
        return
    finetune_directory = os.path.join(STATISTICAL_DIR, "finetune_log")
    if not os.path.exists(finetune_directory):
        os.mkdir(finetune_directory)
    global FINETUNE_STEP
    FINETUNE_STEP += 1
    with open(os.path.join(finetune_directory, f"step_{FINETUNE_STEP}.txt"), "wt") as f:
        f.write(prompt)
        f.write("\n[This is a split string for finetuning]\n")
        f.write(completion)


def complete_text_claude(prompt, stop_sequences=None, model="claude-v1", max_tokens_to_sample = 2000, temperature=0.5, log_file=None, **kwargs):
    raise LLMError("Anthropic backend is disabled in this deployment. Use DeepSeek/OpenAI-compatible models.")


def get_embedding_crfm(text, model="openai/gpt-4-0314"):
    raise LLMError("CRFM backend is disabled in this deployment.")
    
def complete_text_crfm(prompt=None, stop_sequences = None, model="openai/gpt-4-0314",  max_tokens_to_sample=2000, temperature = 0.5, log_file=None, messages = None, **kwargs):
    raise LLMError("CRFM backend is disabled in this deployment.")


def complete_text_openai(prompt, stop_sequences=[], model="gpt-3.5-turbo", max_tokens_to_sample=1000, temperature=0.5, log_file=None, **kwargs):
    # OpenAI-compatible 通道：
    # - 可直连 OpenAI
    # - 也可通过 base_url 指向 DeepSeek / 其他兼容服务
    """ Call the OpenAI API to complete a prompt."""
    raw_request = {
          "model": model,
          "temperature": temperature,
          "max_tokens": max_tokens_to_sample,
          "stop": stop_sequences or None,  # API doesn't like empty list
          **kwargs
    }

    client = _get_openai_client(model)
    openai_module = openai
    
    iteration = 0
    completion = None
    last_error = None
    while iteration < 3:
        try:
            messages = [{"role": "user", "content": prompt}]
            if client is not None:
                try:
                    response = client.chat.completions.create(**{"messages": messages, **raw_request})
                    completion = response.choices[0].message.content
                except Exception:
                    response = client.completions.create(**{"prompt": prompt, **raw_request})
                    completion = response.choices[0].text
            else:
                if openai_module is None:
                    raise LLMError("openai package is not installed")
                try:
                    response = openai_module.ChatCompletion.create(**{"messages": messages, **raw_request})
                    completion = response["choices"][0]["message"]["content"]
                except Exception:
                    response = openai_module.Completion.create(**{"prompt": prompt, **raw_request})
                    completion = response["choices"][0]["text"]

            if completion is not None and not isinstance(completion, str):
                completion = str(completion)
            break
        except Exception as e:
            iteration += 1
            last_error = e
            print(f"===== Retry: {iteration} =====")
            print(f"Error occurs when calling API: {e}")
            continue

    if completion is None:
        raise LLMError(last_error if last_error is not None else "OpenAI-compatible completion failed after retries")
    
    ## Count consumed tokens.
    if STATISTICAL_DIR:
        statitical_file = STATISTICAL_DIR + "/count_tokens.txt"
        try:
            with open(statitical_file, 'r') as file:
                current_token_count = int(file.read().strip())
        except FileNotFoundError:
            current_token_count = 0
        num_prompt_tokens = len(enc.encode(f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"))
        num_sample_tokens = len(enc.encode(completion))
        total_token_count = current_token_count + num_prompt_tokens + num_sample_tokens
        with open(statitical_file, 'w') as file:
            file.write(str(total_token_count))
    
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text(prompt, log_file, model, **kwargs):
    # 统一路由器（本地部署版）：只允许 OpenAI-compatible（建议 deepseek-*）。
    """ Complete text using the specified model with appropriate API. """

    completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    return completion

# specify fast models for summarization etc
FAST_MODEL = "deepseek-chat"
def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)
# complete_text_fast = partial(complete_text_openai, temperature= 0.01)

