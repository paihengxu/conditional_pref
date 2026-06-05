from typing import Optional

def truncate_text(
    text: str,
    max_words: Optional[int] = None,
    max_chars: Optional[int] = None,
    max_tokens: Optional[int] = None,
    truncation_message: str = "[... rest of text is truncated]"
) -> str:
    """
    Truncate text based on word limit, character limit, or token limit (using tiktoken).

    Args:
        text: Input text to truncate
        max_words: Maximum number of words allowed
        max_chars: Maximum number of characters allowed
        max_tokens: Maximum number of tokens allowed (uses cl100k_base tokenizer)
        truncation_message: Suffix added to the text if truncation occurred

    Returns:
        Truncated text string
    """
    if all(x is None for x in [max_words, max_chars, max_tokens]):
        return text

    if text.endswith(truncation_message):
        return text
    
    truncated = text

    # Word-based truncation
    if max_words is not None:
        words = text.split()
        if len(words) > max_words:
            truncated = ' '.join(words[:max_words])

    # Character-based truncation
    if max_chars is not None:
        if len(truncated) > max_chars:
            truncated = truncated[:max_chars]

    # Token-based truncation
    if max_tokens is not None:
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "Token-based truncation requires the 'tiktoken' package.\n"
                "Please run `pip install tiktoken` or install the requirements."
            ) from e
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(truncated)
        if len(tokens) > max_tokens:
            truncated = enc.decode(tokens[:max_tokens])

    if truncated != text:
        truncated += truncation_message

    return truncated

