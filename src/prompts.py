"""
Prompt templates for the Gemini grammatical error correction baseline.
"""


SYSTEM_PROMPT = (
    "You are a grammar correction tool. "
    "Correct grammatical errors and awkward phrasing in the sentence. "
    "Keep the original meaning. "
    "Do not explain your changes. "
    "Output only the corrected sentence."
)


ZERO_SHOT_DETAILED = (
    "Rewrite the following sentence so that it is grammatically correct and "
    "reads fluently. Keep the original meaning and do not add or remove "
    "information. Output only the corrected sentence, with no explanation, "
    "quotation marks, or extra text.\n\n"
    "Sentence: {sentence}\n"
    "Corrected:"
)


FEW_SHOT_INSTRUCTIONS = (
    "Rewrite each sentence so that it is grammatically correct and reads "
    "fluently. Keep the original meaning. Output only the corrected "
    "sentence, with no explanation.\n\n"
)


FEW_SHOT_EXAMPLE_TEMPLATE = (
    "Sentence: {source}\n"
    "Corrected: {target}\n\n"
)


FEW_SHOT_QUERY_TEMPLATE = (
    "Sentence: {sentence}\n"
    "Corrected:"
)


def build_few_shot_prompt(sentence, examples):
    example_block = "".join(
        FEW_SHOT_EXAMPLE_TEMPLATE.format(
            source=source,
            target=target
        )
        for source, target in examples
    )

    return (
        FEW_SHOT_INSTRUCTIONS
        + example_block
        + FEW_SHOT_QUERY_TEMPLATE.format(sentence=sentence)
    )