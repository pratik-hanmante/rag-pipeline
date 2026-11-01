# chunking.py
# ---------------------------------------------------------------------------
# Splits a long document into small pieces ("chunks") for the RAG pipeline.
#
# Why chunk at all?
#   - An LLM can only read a limited number of tokens at once (its context window).
#   - For any one question, only a few paragraphs of a document are relevant.
#   So we cut documents into pieces, embed each piece separately, and at query
#   time fetch only the pieces that match the question. Each chunk becomes one
#   row in the `chunks` table (see schema.sql).
#
# Why "recursive"?
#   We want to cut at NATURAL boundaries so a chunk doesn't end mid-sentence.
#   So we try the "biggest" boundary first and only fall back to smaller ones
#   when a piece is still too big:
#       paragraphs ("\n\n") → lines ("\n") → sentences (". ") → words (" ")
#       → and if even a single word is too long: cut the raw tokens.
# ---------------------------------------------------------------------------

import tiktoken  # OpenAI's tokenizer: counts tokens exactly like the models do


def chunk_recursive(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split `text` into chunks of at most `chunk_size` tokens.

    Args:
        text: the full document text.
        chunk_size: max tokens per chunk. 512 is a common middle ground:
            big enough to hold a full idea, small enough to stay focused.
        overlap: how many tokens each chunk should share with the previous one,
            so a sentence on a boundary isn't lost.
            ⚠️ NOT USED YET: the function accepts it but never applies it.

    Returns:
        A list of chunk strings, in document order. Their positions in the list
        are the `chunk_index` values stored in the database (0, 1, 2…).

    Example:
        >>> chunk_recursive("Para one.\\n\\nPara two.", chunk_size=512)
        ['Para one.\\n\\nPara two.']    # fits in one chunk, returned whole
    """
    # Boundaries to split on, from best (keeps the most meaning together)
    # to worst. The function works down this list only when it has to.
    separators = ["\n\n", "\n", ". ", " "]

    # The tokenizer. "cl100k_base" is the encoding used by OpenAI's
    # text-embedding-3-* and GPT-4-era models, so our counts match theirs.
    # (Downloaded once on first use, then cached on disk.)
    enc = tiktoken.get_encoding("cl100k_base")

    # Inner helper that calls itself ("recursion"). It splits `text` using the
    # first separator in `seps`, and any piece that's still too big is split
    # again using the remaining, smaller separators.
    def _split(text: str, seps: list[str]) -> list[str]:
        # Case 1: already small enough, so keep it as one chunk.
        # len(enc.encode(text)) = number of tokens in the text.
        if len(enc.encode(text)) <= chunk_size:
            return [text]

        # Case 2: too big and NO separators left (e.g. one enormous "word"
        # like a long URL or base64 blob). Last resort: cut the token list
        # into fixed-size slices and turn each slice back into text.
        #   tokens[0:512], tokens[512:1024], ...
        if not seps:
            tokens = enc.encode(text)
            return [enc.decode(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

        # Case 3: too big, but we still have separators to try.
        # Take the current separator; keep the smaller ones for later.
        sep, rest_seps = seps[0], seps[1:]
        # e.g. sep="\n\n" → list of paragraphs
        parts = text.split(sep)

        # Greedy packing: keep adding parts to `buffer` while it still fits.
        # When the next part would overflow, the buffer is "full": save it
        # and start a new buffer with that part.
        results, buffer = [], ""
        for part in parts:
            # What the buffer would look like if we added this part.
            # (Re-insert the separator we removed with .split().)
            candidate = buffer + sep + part if buffer else part

            if len(enc.encode(candidate)) <= chunk_size:
                # Still fits, so keep growing the buffer.
                buffer = candidate
            else:
                # Doesn't fit: save what we have so far…
                if buffer:
                    # …running it through _split with smaller separators.
                    # Normally it already fits and comes back unchanged.
                    results.extend(_split(buffer, rest_seps))
                # …and start fresh with the part that didn't fit. If this
                # single part is itself too big, the next save (or the final
                # one below) will break it down with the smaller separators.
                buffer = part

        # Don't forget whatever is left in the buffer after the loop.
        if buffer:
            results.extend(_split(buffer, rest_seps))
        return results

    # Start with the full list of separators (paragraphs first).
    return _split(text, separators)
