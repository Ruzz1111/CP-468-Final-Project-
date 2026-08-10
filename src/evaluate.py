import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from nltk.translate.gleu_score import corpus_gleu
from torch.utils.data import DataLoader, Dataset

from dataset import GrammarErrorCorrectionDataset
from model import EncoderLSTM, DecoderLSTM, Seq2Seq


class SourceDataset(Dataset):
    def __init__(self, gec_dataset, max_len):
        self.gec_dataset = gec_dataset
        self.max_len = max_len

    def __len__(self):
        return len(self.gec_dataset.data)

    def __getitem__(self, idx):
        sentence = self.gec_dataset.data.iloc[idx]["sentence"]
        tokens = self.gec_dataset.tokenize(sentence)
        ids = self.gec_dataset.encode(tokens, self.max_len)
        return torch.tensor(ids, dtype=torch.long), idx


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    word2idx = checkpoint["word2idx"]
    idx2word = checkpoint["idx2word"]
    vocab_size = len(word2idx)

    encoder = EncoderLSTM(
        vocab_size,
        checkpoint["embed_size"],
        checkpoint["hidden_size"],
    )
    decoder = DecoderLSTM(
        vocab_size,
        checkpoint["embed_size"],
        checkpoint["hidden_size"],
    )
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, word2idx, idx2word


@torch.no_grad()
def greedy_decode(model, source, word2idx, max_len):
    sos_idx = word2idx["<SOS>"]
    eos_idx = word2idx["<EOS>"]
    pad_idx = word2idx["<PAD>"]

    encoder_states, hidden, cell = model.encoder(source)
    current = torch.full(
        (source.size(0),), sos_idx, dtype=torch.long, device=source.device
    )

    generated = [[] for _ in range(source.size(0))]
    finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)

    for _ in range(max_len - 1):
        output, hidden, cell = model.decoder(
            current, hidden, cell, encoder_states
        )
        current = output.argmax(dim=1)

        for i, token_id in enumerate(current.tolist()):
            if finished[i]:
                continue
            if token_id == eos_idx:
                finished[i] = True
            elif token_id != pad_idx:
                generated[i].append(token_id)

        if finished.all():
            break

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--test-csv", default="dataset/test.csv")
    parser.add_argument("--output", default="lstm_test_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, word2idx, idx2word = load_checkpoint(args.checkpoint, device)

    test_gec = GrammarErrorCorrectionDataset(args.test_csv)
    test_gec.word2idx = word2idx.copy()
    test_gec.idx2word = idx2word.copy()

    test_data = SourceDataset(test_gec, checkpoint["max_len"])
    if args.limit is not None:
        test_data = torch.utils.data.Subset(
            test_data, range(min(args.limit, len(test_data)))
        )

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    predictions = {}
    for source, row_indices in test_loader:
        source = source.to(device)
        generated = greedy_decode(
            model, source, word2idx, checkpoint["max_len"]
        )
        for row_idx, token_ids in zip(row_indices.tolist(), generated):
            predictions[row_idx] = [
                idx2word.get(token_id, "<UNK>") for token_id in token_ids
            ]

    references = []
    hypotheses = []
    rows = []

    for row_idx in sorted(predictions):
        row = test_gec.data.iloc[row_idx]
        reference_texts = json.loads(row["corrections"])
        reference_tokens = [test_gec.tokenize(text) for text in reference_texts]
        hypothesis_tokens = predictions[row_idx]

        references.append(reference_tokens)
        hypotheses.append(hypothesis_tokens)
        rows.append(
            {
                "source": row["sentence"],
                "references": json.dumps(reference_texts, ensure_ascii=False),
                "lstm_output": " ".join(hypothesis_tokens),
            }
        )

    gleu = corpus_gleu(references, hypotheses) if rows else 0.0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Device: {device}")
    print(f"Test examples: {len(rows)}")
    print(f"GLEU: {gleu:.4f}")
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
