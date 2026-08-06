from datasets import load_dataset
import pandas as pd
import os

def fetch_gec_data():
    
    dataset = load_dataset("jfleg")
    
    
    
    os.makedirs("dataset", exist_ok=True)
    
   
    dataset['validation'].to_csv("dataset/validation.csv", index=False)##creating dataset files 
    dataset['test'].to_csv("dataset/test.csv", index=False)
    
    print("saved to the /dataset directory.")

if __name__ == "__main__":
    fetch_gec_data()