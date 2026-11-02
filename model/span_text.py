# coding:utf-8
"""Recover an entity's surface form from the source indices a model predicted.

The decoder emits positions in the BPE sequence. Earlier code turned those back
into text by decoding the sub-token each position pointed at, which prints only
the first sub-token of a word. A character such as 嶽 is tokenised as a bare
word-start marker followed by the character, so that marker is all the printed
string kept, and the character was lost. Mapping each position back to the word
that contains it recovers the original characters instead.
"""
import bisect


def word_cum_lens(words, tokenizer):
    """BPE start offset of every word, matching data_pipe.get_tokens_from_words"""
    cum, n = [], 1  # position 0 holds the bos token
    for w in words:
        cum.append(n)
        n += len(tokenizer.tokenize(w))
    cum.append(n)
    return cum


def span_text(indices, words, cum, word_start_index):
    out, last = [], -1
    for kk in indices:
        pos = kk - word_start_index
        w = bisect.bisect_right(cum, pos) - 1
        if 0 <= w < len(words) and w != last:
            out.append(words[w])
            last = w
    return "".join(out)
