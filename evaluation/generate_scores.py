import json

import pandas as pd

import argparse

def main():
    parse = argparse.ArgumentParser(description="Summarize Evaluation Metrics")
    parse.add_argument(
        "--input_file", type=str, default="/root/nfs/hmj/proj/mem0/evaluation/src/memMini/results/evaluation_metrics.json", help="Path to the evaluation metrics file"
    )
    args = parse.parse_args()
    
    # Load the evaluation metrics data
    with open(args.input_file, "r") as f:
        data = json.load(f)

    # Flatten the data into a list of question items
    all_items = []
    for key in data:
        all_items.extend(data[key])

    # Convert to DataFrame
    df = pd.DataFrame(all_items)

    # Convert category to numeric type
    df["category"] = pd.to_numeric(df["category"])

    # Calculate mean scores by category
    result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    # Add count of questions per category
    result["count"] = df.groupby("category").size()

    # Print the results
    print("Mean Scores Per Category:")
    print(result)

    # Calculate overall means
    overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    print("\nOverall Mean Scores:")
    print(overall_means)
if __name__ == "__main__":
    main()