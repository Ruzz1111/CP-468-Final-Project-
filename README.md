# CP-468-Final-Project-
The final Seq2Seq AI project 

## LSTM Training and Evaluation

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Train the LSTM model:

```bash
python src/train.py --epochs 10 --batch-size 32 --embed-size 256 --hidden-size 256 --max-len 50 --output best_model.pt
```

Evaluate the trained model on the test set:

```bash
python src/evaluate.py --checkpoint best_model.pt --test-csv dataset/test.csv --output lstm_test_predictions.csv
```

For our final run, the model was trained for 10 epochs and evaluated on 748 test examples. The LSTM achieved a GLEU score of 0.2348.
