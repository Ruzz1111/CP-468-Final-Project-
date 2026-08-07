from datasets import load_dataset
import pandas as pd
import os
import json

def fetch_gec_data():
    
    dataset = load_dataset("jfleg")
    
    
    
    os.makedirs("dataset", exist_ok=True)
    #create both validation and test
    df = dataset['validation'].to_pandas()
    corrections_json = []
    for refs in df["corrections"]:
        corrections_json.append(json.dumps(list(refs)))
    df["corrections"] = corrections_json
    df.to_csv("dataset/validation.csv", index=False)

    df = dataset['test'].to_pandas()
    corrections_json = []
    for refs in df["corrections"]:
        corrections_json.append(json.dumps(list(refs)))
    df["corrections"] = corrections_json
    df.to_csv("dataset/test.csv", index=False)
    
    
    print("saved to the /dataset directory.")

def fetch_train_data():

    dataset = load_dataset("agentlans/grammar-correction")
    os.makedirs("dataset", exist_ok=True)
    #create training set
    df = dataset['train'].to_pandas()
    df = df.sample(n=10000, random_state=0)

    train_df = pd.DataFrame({
        "sentence": df["input"],
        "corrections": df["output"],
    })

    corrections_json = []
    for correction in train_df["corrections"]:

        corrections_json.append(json.dumps([correction]))

    train_df["corrections"] = corrections_json
 
    train_df.to_csv("dataset/train.csv", index=False)
 
    print(f"saved {len(train_df)} rows to dataset/train.csv")

if __name__ == "__main__":
    fetch_gec_data()
    fetch_train_data()