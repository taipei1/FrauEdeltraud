"""
translator.py — Rewrites English text using only the learner's known vocabulary.

Uses the LLM to intelligently replace unknown words with known equivalents
while preserving meaning. The vocabulary list comes from the PostgreSQL database.
"""
import os
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

log = logging.getLogger("translator")


def translate_with_vocabulary(
    text: str,
    known_words: list[str],
    max_vocab_in_prompt: int = 500,
) -> str:
    """Rewrite text using only known vocabulary words.
    
    Args:
        text: The English text to simplify.
        known_words: List of English words/phrases the learner knows.
        max_vocab_in_prompt: Max number of words to include in the prompt.
    
    Returns:
        Rewritten text using known vocabulary.
    """
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    
    vocab_str = ", ".join(known_words[:max_vocab_in_prompt])
    
    system_prompt = f"""You are a vocabulary simplifier for an English learner at CEFR A2 level.

Your task: rewrite the given English text so that it uses ONLY words from the learner's KNOWN VOCABULARY list below, plus basic grammar words (a, the, is, are, was, were, I, you, he, she, it, we, they, this, that, and, or, but, not, no, yes, very, much, many, some, any, all, do, does, did, have, has, had, can, could, will, would, to, in, on, at, for, with, from, by, of, about, up, down, out, off, if, when, then, so, because, also, too, more, less, most, just, only, still, already, now, here, there, how, what, where, who, why, which, my, your, his, her, its, our, their, me, him, us, them, one, two, three, first, last, new, old, good, bad, big, small, long, short, well, really, thing, time, day, way, people, man, woman, make, go, come, get, know, think, say, see, want, give, take, look, put, work, try, use, need, feel, tell, ask, seem, help, show, turn, play, run, move, like, live, mean, keep, let, begin, start, end, stop, open, close, read, write, learn, speak, eat, drink, sleep, sit, stand, walk, talk, call, pay, buy, sell, send, hold, bring, meet, hear, spend, grow, set, kind, part, number, hand, place, case, point, group, eye, fact, world, child, year, month, week, side, room, head, face, word, life, house, school, home, back, door, water, food, name, city, car, game, book, job).

KNOWN VOCABULARY:
{vocab_str}

RULES:
1. Replace unknown words with the closest KNOWN word or short phrase from the list
2. Keep the SAME meaning as the original
3. If no good replacement exists, use a simple A2-level explanation (2-3 basic words)
4. Keep sentences short and simple
5. Output ONLY the rewritten text, nothing else
6. Do NOT add explanations, comments, or notes"""

    llm = ChatGroq(
        model=model_name,
        temperature=0.3,
        max_tokens=500,
        groq_api_key=api_key,
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Rewrite this text:\n{text}"),
    ]
    
    response = llm.invoke(messages)
    result = response.content.strip()
    log.info("Translated: %r -> %r", text[:80], result[:80])
    return result
