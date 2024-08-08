import json
import openai
from openai import OpenAI
import tiktoken
import time
import copy

from prompt import ROLE, PROMPT
from Config import API_KEY

MAX_TOKENS = 4097
OUTPUT_TOKENS = 300

client = OpenAI(
  api_key=API_KEY,  # this is also the default, it can be omitted
)

def num_tokens_from_messages(messages, model="gpt-4"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_model(messages):
    tokens = num_tokens_from_messages(messages)
    if tokens < MAX_TOKENS - OUTPUT_TOKENS:
        model = "gpt-3.5-turbo"
    else:
        print("Using 16k model")
        model = "gpt-3.5-turbo-16k"

    return model


def ask_gpt(appName, history, view, task, previous_xml=None):

    hint = ""
    if view == previous_xml:
        # last_action = formatted_history[-1]
        # ll_action =  last_action["action"]
        # ll_direction = last_action.get("direction", "")
        # ll_id = last_action.get("id", "") + last_action.get("scroll-reference", "")
        hint = f"\n Do not repeat the last action in history actions"
    formatted_history = copy.deepcopy(history)
    # formatted_history = history.copy() will affect the content of histroy
    for action in formatted_history:
        if "time_view" in action:
            del action["time_view"]
        if "time_process_vh" in action:
            del action["time_process_vh"]
        if "time_askGPT" in action:
            del action["time_askGPT"]
        if "time_performAction" in action:
            del action["time_performAction"]
    formatted_history = "\n".join(json.dumps(h) for h in formatted_history) if len(formatted_history) > 0 else "None"

    if "scroll-reference" not in view:
        # print("Removing scroll action")
        role_template = "\n".join(line for line in ROLE.split("\n") if "scroll" not in line)
    else:
        role_template = ROLE
    currPrompt  = PROMPT.format(appName, task, formatted_history, view)
    currPrompt += hint
    messages = [
        {"role": "system", "content": role_template},
        {"role": "user", "content": currPrompt},
    ]
    print("\n\n==== Prompts =====")
    # print("### SYSTEM")
    # print(messages[0]["content"])

    print("\n### USER")
    print(messages[1]["content"])

    # model = get_model(messages)
    model = "gpt-4"

    print("Getting ChatGPT response")
    response = get_chat_completion(model=model, messages=messages)
    
    print("\n### RESPONSE")
    print(response)

    return response


def get_chat_completion(**kwargs):
    while True:
        try:
            return client.chat.completions.create(**kwargs, temperature=0.2)
        except openai.RateLimitError as e:
            print(e)
            # Waiting 1 minute as that is how long it takes the rate limit to reset
            print("Rate limit reached, waiting 1 minute")
            time.sleep(60)

