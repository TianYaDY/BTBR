from typing import List, Tuple, TypedDict

class Message(TypedDict):
    role: str  # roles are 'system', 'user', 'assistant'
    content: str

Dialog = List[Message]

def construct_prompt(dialog: Dialog) -> str:
    prompt_segments = ["<|begin_of_text|>"]

    for message in dialog:
        role = message["role"]
        content = message["content"]


        if role == "system":
            prompt_segments.append(f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>")
        elif role == "user" or role == "assistant":
            prompt_segments.append(f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>")
        else:
            raise ValueError("Unsupported role")


    if dialog[-1]["role"] != "user":
        raise ValueError("Last message must be from user")


    prompt_segments.append("<|start_header_id|>assistant<|end_header_id|>")

    return "".join(prompt_segments)


if __name__ == "__main__":

    dialog_example = [
        {"role": "system", "content": "You are a helpful AI assistant.\n"},
        {"role": "user", "content": "What is the capital of France?\n"},
        {"role": "assistant", "content": "The capital of France is Paris.\n"},
        {"role": "user", "content": "Thank you!\n"}
    ]
    prompt = construct_prompt(dialog_example)
    print(prompt)
