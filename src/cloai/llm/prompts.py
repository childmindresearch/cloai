# ruff: noqa: E501
"""Stores non-parameterized prompts used by the LLM."""

import string
from collections.abc import Iterable


def _remove_consecutive_whitespace(text: str) -> str:
    """Removes consecutive whitespaces from a string.

    All consecutive whitespaces are replaced with a single space.

    Args:
        text: The text to remove consecutive whitespaces from.

    Returns:
        The text without consecutive whitespaces.
    """
    return " ".join(text.split())


def _substitute(text: str, **kwargs: str) -> str:
    """Substitutes ${var} templates in the input.

    Args:
        text: The text to substitute.
        kwargs: Keyword arguments to substitute.

    Returns:
        The substituted text.
    """
    template = string.Template(text)
    identifier = template.get_identifiers()
    if excess_keys := set(kwargs.keys()) - set(identifier):
        msg = f"Too many keys provided: {excess_keys}."
        raise ValueError(msg)

    return template.substitute(**kwargs)


def chain_of_verification_create_statements() -> str:
    """Chain of Verification statement creation prompt.

    Returns:
        The Prompt.
    """
    return _remove_consecutive_whitespace(
        """
            Based on the following instructions, write a set of statements that can be
            answered with True or False to determine whether a piece of text adheres to
            these instructions. True should denote adherence to the structure whereas
            False should denote a lack of adherence.
        """,
    )


def chain_of_verification_verify(statements: Iterable[str]) -> str:
    """Chain of verification statement verification prompt.

    Args:
        statements: Statements to verify.

    Returns:
        The prompt.
    """
    return _substitute(
        _remove_consecutive_whitespace(
            """
                Based on the following statements and the text provided by the user, return True or False to assess whether the statements
                are correct or not. The statements are as follows:
                ${statements}.
            """,
        ),
        statements="\n\n".join(statements),
    )


def chain_of_verification_rewrite(
    statements: Iterable[str],
    instructions: str,
    source: str,
) -> str:
    """Chain of verification statement verification prompt.

    Args:
        statements: Statements to verify.
        instructions: The original instructions.
        source: The original source material.

    Returns:
        The prompt.
    """
    return _substitute(
        """
Based on the following statements and source material, rewrite the user provided text such that
all statements can be answered with 'True'. The statements are as follows:

${statements}

The original source material is material is: ${source}

The original instructions were: ${instructions}
""",
        statements="\n\n".join(statements),
        instructions=instructions,
        source=source,
    )


def chain_of_density(article: str) -> str:
    """Chain of density prompt.

    Args:
        article: The article to summarize.

    Returns:
        The prompt.
    """
    return _substitute(
        """Article: ${article}

You will generate increasingly concise, entity-dense summaries of the above Article.

Step 1. Identify 1-3 informative Entities from the Article which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

A Missing Entity is:
- Relevant: to the main story.
- Specific: descriptive yet concise (5 words or fewer).
- Novel: not in the previous summary.
- Faithful: present in the Article.
- Anywhere: located anywhere in the Article.

Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

Remember, use the exact same number of words for each summary.
""",
        article=article,
    )
