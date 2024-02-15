import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def calc_topk_accuracy(preds, labels, k=5):
    """
    Computes top-k accuracy, from predicted and true label names.
    """
    assert len(preds) == len(labels)
    assert k > 0
    assert k <= len(preds[0])
    correct = 0
    for i in range(len(preds)):
        if labels[i] in preds[i][:k]:
            correct += 1
    return correct / len(preds)


def binary_classification_correctness(csv_data: str, categories: list = None) -> dict():
    """
    Calculate binary classification correctness for each category, as well as multi-class correctness.
    """
    # check if DataFrame, else load a file:
    if isinstance(csv_data, pd.DataFrame):
        df = csv_data
    else:
        # Load the CSV file
        df = pd.read_csv(csv_data)

    # List of categories
    if categories is None:
        # Exclude non-category columns to get the list of categories
        non_category_columns = ['filename', 'gt', 'pred', 'score']
        categories = [col for col in df.columns if col not in non_category_columns]

    # Calculate the classification correctness for each category
    binary_results = {}
    for category in categories:
        # Create a binary ground truth column: 1 if gt matches category, 0 otherwise
        df['binary_gt'] = (df['gt'] == category).astype(int)

        # Create a binary prediction column: 1 if pred matches category, 0 otherwise
        df['binary_pred'] = (df['pred'] == category).astype(int)

        # Calculate binary correctness
        correct_predictions = (df['binary_gt'] == df['binary_pred']).sum()
        total_predictions = len(df)

        binary_correctness = correct_predictions / total_predictions
        binary_results[category] = binary_correctness

    # Calculate multi-class correctness
    multi_class_correct = (df['gt'] == df['pred']).sum()
    multi_class_correctness = multi_class_correct / len(df)

    # Print results
    output_prints = []
    for category, correctness in binary_results.items():
        output_prints.append(f"Binary correctness for {category}: {correctness:.2f}")

    output_prints.append(f"\nMulti-class correctness: {multi_class_correctness:.2f}")

    return binary_results, multi_class_correctness, output_prints


def build_confision_matrix(csv_data: str, base_dir: str = None):
    """ Build a confusion matrix from a CSV file. """
    # check if DataFrame, else load a file:
    if isinstance(csv_data, pd.DataFrame):
        df = csv_data
    else:
        # Load the CSV file
        df = pd.read_csv(csv_data)
        base_dir = os.path.dirname(path_to_csv_file)

    # Build confusion matrix:
    # -----------------------
    # Exclude non-category columns to get the list of categories
    non_category_columns = ['filename', 'gt', 'pred', 'score']
    categories = [col for col in df.columns if col not in non_category_columns]

    # Generate the confusion matrix
    confusion_matrix = pd.crosstab(df['gt'], df['pred'], rownames=['Actual'], colnames=['Predicted'])

    # Plotting the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt="d")

    plt.title("Confusion Matrix")
    #plt.show()
    plt.tight_layout()  # Adjusts the layout to ensure axis titles aren't cut off
    plt.savefig(os.path.join(base_dir, "confusion_matrix.png"))

    # several metrics you can derive from a confusion matrix to quantify the quality of classification
    import numpy as np

    # Ensure confusion_matrix has all categories for both rows and columns
    # --------------------------------------------------------------------
    all_categories = df['gt'].unique().tolist() + df['pred'].unique().tolist()
    all_categories = list(set(all_categories))  # Removing duplicates
    confusion_matrix = pd.crosstab(df['gt'], df['pred'], rownames=['Actual'], colnames=['Predicted']).reindex(
        index=all_categories, columns=all_categories, fill_value=0)

    total = np.sum(confusion_matrix.to_numpy())
    correct_predictions = np.trace(confusion_matrix.to_numpy())
    accuracy = correct_predictions / total

    # For multi-class, we'll compute macro averages for precision, recall, and F1-score
    precisions = []
    recalls = []
    f1_scores = []

    for category in all_categories:
        TP = confusion_matrix.at[category, category]
        FP = confusion_matrix[category].sum() - TP
        FN = confusion_matrix.loc[category].sum() - TP
        TN = total - TP - FP - FN

        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    # Macro averages
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    results = {
        "accuracy": accuracy,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1
    }

    output_prints = []
    output_prints.append(f"Accuracy: {accuracy:.2f}")
    output_prints.append(f"Macro Precision: {avg_precision:.2f}")
    output_prints.append(f"Macro Recall: {avg_recall:.2f}")
    output_prints.append(f"Macro F1-Score: {avg_f1:.2f}")

    # Identify confusing pairs:
    # -------------------------
    # Find off-diagonal values and sort them in descending order
    confusing_pairs = {}
    for i in all_categories:
        for j in all_categories:
            if i != j:
                confusing_pairs[(i, j)] = confusion_matrix.at[i, j]

    # Sort the pairs by confusion in descending order
    sorted_confusing_pairs = sorted(confusing_pairs.items(), key=lambda x: x[1], reverse=True)

    # Print the top N confusing pairs. Change top_n as required.
    top_n = 5  # Adjust this number as required
    for idx, (pair, count) in enumerate(sorted_confusing_pairs[:top_n]):
        output_prints.append(f"#{idx + 1}: Classes {pair[0]} and {pair[1]} were confused {count} times.")

    return results, confusion_matrix, output_prints

# main
if __name__ == '__main__':
    path_to_csv_file = "/home/guy/code/grounding_tmp/CUB/tiny5/BaseProgramClassifier/ViT_L_14_336px/ox_prompt_noname4/files_programs_CUB_gpt4_prompt_program_guybase_/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4_ViT-L-14-336px_results.csv"
    # categories = ['Least Auklet', 'Heermann Gull', 'Laysan Albatross', 'Rhinoceros Auklet', 'Sooty Albatross']
    # path_to_csv_file = "/shared-data5/guy/exps/grounding_logs/CUB/tiny/BaseProgramClassifier/ViT_L_14_336px/ox_prompt_noname4/files_programs_CUB_gpt4_prompt_program_guybase_/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4_ViT-L-14-336px_results.csv"
    # path_to_csv_file = "/shared-data5/guy/exps/grounding_logs/CUB/full/BaseAttributeClassifier/ViT_L_14_336px/ox_prompt_noname4/cub_gpt3_text-davinci-003_descriptions_ox_prompt_noname4_ViT-L-14-336px_results.csv"
    categories = None
    binary_results, multi_class_correctness, out_prints = binary_classification_correctness(path_to_csv_file, categories=categories)
    [print(x) for x in out_prints]
    print(binary_results)
    results, sorted_confusing_pairs, out_prints = build_confision_matrix(path_to_csv_file)
    [print(x) for x in out_prints]
    print(results)
    print(sorted_confusing_pairs)




