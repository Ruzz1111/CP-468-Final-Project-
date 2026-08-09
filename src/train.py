import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GrammarErrorCorrectionDataset, GECDatasetWrapper
from model import EncoderLSTM, DecoderLSTM, Seq2Seq


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_vocab(source, target):
    target.word2idx = source.word2idx.copy()
    target.idx2word = source.idx2word.copy()


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_num, (source, target) in enumerate(loader):
        if max_batches is not None and batch_num >= max_batches:
            break

        source = source.to(device)
        target = target.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            output = model(source, target, teacher_forcing_ratio=0.5 if training else 0.0)
            vocab_size = output.shape[-1]
            loss = criterion(
                output[:, 1:, :].reshape(-1, vocab_size),
                target[:, 1:].reshape(-1),
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--output", default="best_model.pt")
    args = parser.parse_args()

    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_gec = GrammarErrorCorrectionDataset("dataset/train.csv")
    train_gec.build_vocab(min_count=3)

    val_gec = GrammarErrorCorrectionDataset("dataset/validation.csv")
    copy_vocab(train_gec, val_gec)

    train_data = GECDatasetWrapper(train_gec, max_len=args.max_len)
    val_data = GECDatasetWrapper(val_gec, max_len=args.max_len)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    vocab_size = len(train_gec.word2idx)
    encoder = EncoderLSTM(vocab_size, args.embed_size, args.hidden_size)
    decoder = DecoderLSTM(vocab_size, args.embed_size, args.hidden_size)
    model = Seq2Seq(encoder, decoder, device).to(device)

    pad_idx = train_gec.word2idx["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Trainable parameters: {parameter_count:,}")

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer, args.max_train_batches
        )
        val_loss = run_epoch(
            model, val_loader, criterion, device, None, args.max_val_batches
        )

        print(f"Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "word2idx": train_gec.word2idx,
                    "idx2word": train_gec.idx2word,
                    "embed_size": args.embed_size,
                    "hidden_size": args.hidden_size,
                    "max_len": args.max_len,
                },
                args.output,
            )
            print(f"Saved best model to {args.output}")

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
