import tiktoken

def count_num_tokens(text: str, model: str) -> int:
    """
    Returns the number of tokens in the given text.
    Args:
        text (str): The text to count tokens in.
        model (str): The name of the LLM model.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unsupported models like LLaMA, Mistral, etc.
        print(f"[Warning] Model '{model}' not supported by tiktoken. Using 'cl100k_base' as fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
