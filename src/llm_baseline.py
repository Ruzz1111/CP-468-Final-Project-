import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from nltk.translate.gleu_score import corpus_gleu

from src.dataset import GrammarErrorCorrectionDataset

from src.prompts import (
    SYSTEM_PROMPT,
    ZERO_SHOT_DETAILED,
    build_few_shot_prompt,
)


def load_few_shot_examples(train_csv, k=4, seed=42):
    df = pd.read_csv(train_csv)

    sample = df.sample(
        n=k,
        random_state=seed
    )

    examples = []

    for _, row in sample.iterrows():
        corrections = json.loads(
            row["corrections"]
        )

        examples.append(
            (
                row["sentence"],
                corrections[0]
            )
        )

    return examples


def build_prompt(
    variant,
    sentence,
    few_shot_examples=None
):
    if variant == "zero_shot_detailed":
        return ZERO_SHOT_DETAILED.format(
            sentence=sentence
        )

    if variant == "few_shot":
        if few_shot_examples is None:
            raise ValueError(
                "Few-shot examples are required."
            )

        return build_few_shot_prompt(
            sentence,
            few_shot_examples
        )

    raise ValueError(
        f"Unknown variant: {variant}"
    )


def call_gemini(
    client,
    model,
    user_prompt,
    max_retries=3
):
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=150,
                ),
            )

            if response.text is None:
                return ""

            return response.text.strip()

        except Exception as error:
            last_error = error

            print(
                f"API error, retry "
                f"{attempt + 1}/{max_retries}: "
                f"{error}"
            )

            time.sleep(2 ** attempt)

    raise last_error


def run_variant(
    client,
    model,
    variant,
    sentences,
    output_path,
    few_shot_examples=None,
):
    rows = []

    start_time = time.time()

    for i, sentence in enumerate(sentences):
        prompt = build_prompt(
            variant,
            sentence,
            few_shot_examples
        )

        try:
            output_text = call_gemini(
                client,
                model,
                prompt
            )

        except Exception as error:
            print()
            print(
                f"Stopped at example "
                f"{i + 1}/{len(sentences)}"
            )

            print(
                f"Error: {error}"
            )

            pd.DataFrame(
                rows
            ).to_csv(
                output_path,
                index=False
            )

            print(
                f"Partial results saved to: "
                f"{output_path}"
            )

            raise

        rows.append({
            "source": sentence,
            "llm_output": output_text
        })

        # Save progress every 10 examples
        if (i + 1) % 10 == 0:
            pd.DataFrame(
                rows
            ).to_csv(
                output_path,
                index=False
            )

            print(
                f"{i + 1}/{len(sentences)} "
                f"completed and saved"
            )

        else:
            print(
                f"{i + 1}/{len(sentences)} "
                f"completed"
            )

    elapsed = (
        time.time() - start_time
    )

    # Final save after successful completion
    pd.DataFrame(
        rows
    ).to_csv(
        output_path,
        index=False
    )

    return rows, elapsed


def score_gleu(
    test_gec,
    rows
):
    references = []
    hypotheses = []

    for output_row, (_, data_row) in zip(
        rows,
        test_gec.data.iterrows()
    ):
        reference_texts = json.loads(
            data_row["corrections"]
        )

        references.append([
            test_gec.tokenize(reference)
            for reference in reference_texts
        ])

        hypotheses.append(
            test_gec.tokenize(
                output_row["llm_output"]
            )
        )

    return corpus_gleu(
        references,
        hypotheses
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-csv",
        default="dataset/test.csv"
    )

    parser.add_argument(
        "--train-csv",
        default="dataset/train.csv"
    )

    parser.add_argument(
        "--output-dir",
        default="results"
    )

    parser.add_argument(
        "--model",
        default="gemini-3.5-flash"
    )

    parser.add_argument(
        "--variant",
        choices=[
            "zero_shot_detailed",
            "few_shot"
        ],
        required=True
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    args = parser.parse_args()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set."
        )

    client = genai.Client(
        api_key=api_key
    )

    test_gec = (
        GrammarErrorCorrectionDataset(
            args.test_csv
        )
    )

    sentences = (
        test_gec.data["sentence"]
        .tolist()
    )

    if args.limit is not None:
        sentences = sentences[
            :args.limit
        ]

    few_shot_examples = None

    if args.variant == "few_shot":
        few_shot_examples = (
            load_few_shot_examples(
                args.train_csv,
                k=args.k,
                seed=42
            )
        )

        print(
            f"Loaded "
            f"{len(few_shot_examples)} "
            f"few-shot examples."
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_model_name = (
        args.model
        .replace("/", "_")
        .replace(":", "_")
    )

    predictions_path = (
        output_dir
        / (
            f"{safe_model_name}_"
            f"{args.variant}_"
            f"predictions.csv"
        )
    )

    print(
        f"Model: {args.model}"
    )

    print(
        f"Variant: {args.variant}"
    )

    print(
        f"Examples: {len(sentences)}"
    )

    print(
        f"Saving progress to: "
        f"{predictions_path}"
    )

    rows, elapsed = run_variant(
        client=client,
        model=args.model,
        variant=args.variant,
        sentences=sentences,
        output_path=predictions_path,
        few_shot_examples=(
            few_shot_examples
        )
    )

    gleu = score_gleu(
        test_gec,
        rows
    )

    summary_path = (
        output_dir
        / (
            f"{safe_model_name}_"
            f"{args.variant}_"
            f"summary.csv"
        )
    )

    average_seconds = (
        elapsed / len(rows)
        if len(rows) > 0
        else 0
    )

    summary = pd.DataFrame([
        {
            "model": args.model,
            "variant": args.variant,
            "examples": len(rows),
            "gleu": round(
                gleu,
                4
            ),
            "runtime_seconds": round(
                elapsed,
                2
            ),
            "average_seconds_per_example":
                round(
                    average_seconds,
                    4
                )
        }
    ])

    summary.to_csv(
        summary_path,
        index=False
    )

    print()

    print(
        f"GLEU: {gleu:.4f}"
    )

    print(
        f"Runtime: "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Average time per example: "
        f"{average_seconds:.4f} seconds"
    )

    print(
        f"Saved predictions to: "
        f"{predictions_path}"
    )

    print(
        f"Saved summary to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()