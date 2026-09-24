"""Prompt template utilities.

Provides a template that includes Role, Context, Task, Format, Length, a
negative constraint, and a small few-shot example. The template is a plain
Python format string that can be filled by the calling code.
"""

PROMPT_TEMPLATE = """
Role: {role}

Context:
{context}

Task:
{task}

Format:
{format}

Length:
{length}

Do not answer using information not in the context.

Example:
Q: "How long does delivery take for an order within serviceable pin codes?"
A: "Delivery typically takes 10 to 30 minutes depending on delivery zone and current order volume."

Now answer the following.
Q: {query}
"""
