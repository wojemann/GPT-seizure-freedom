import os
import json
import csv
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from IPython.display import clear_output

# Utilities

def display_conf_matrix(true, pred, title):
    conf_matrix = confusion_matrix(true, pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def display_conf_matrix_fig(true, pred, title, save_location):
    conf_matrix = confusion_matrix(true, pred)
    sns.set(font_scale=1.6)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig(f'figures/{save_location}', dpi=100)

def display_f1(true, pred):
    f1 = f1_score(true, pred, average=None)
    weighted_f1 = f1_score(true, pred, average='weighted')
    for i, score in enumerate(f1):
        print(f"Class {i}: F1 Score = {round(score, 2)}")
    print(f"Weighted Average F1 Score: {round(weighted_f1, 2)}")

def extract_label_from_text(pred_str):
    labels = []
    if "Yes" in pred_str:
        labels.append(1)
    if "No" in pred_str:
        labels.append(0)
    if "Unclear" in pred_str:
        labels.append(2)
    return labels[0] if labels else 2

def eval_regex(file_name, name, fig_name, labels_file=None):
    true_label = []
    predicted_label = []
    correct = 0

    if labels_file:
        with open(labels_file, 'r') as fp:
            for line in fp:
                true_label.append(json.loads(line)['label'])

    with open(file_name, 'r') as fp:
        for i, line in enumerate(fp):
            data = json.loads(line)
            pred_str = data['pred_str'].split('[/INST]')[-1] if '[INST]' in data['pred_str'] else data['pred_str']
            label = extract_label_from_text(pred_str)

            if not labels_file:
                true_label.append(data['label'])

            predicted_label.append(label)

            if true_label[i] == label:
                correct += 1

    print(f"Correct: {correct}\t\t Accuracy: {round(correct / len(true_label), 2)}")
    display_f1(true_label, predicted_label)
    if fig_name:
        display_conf_matrix_fig(true_label, predicted_label, name, fig_name)
    else:
        display_conf_matrix(true_label, predicted_label, name)

def eval_cot(file_name, name, class_model_output_dir, fig_name,
             temp_file,
             model_path):
    
    if not os.path.exists(class_model_output_dir):
        os.makedirs(class_model_output_dir)

    true_labels = []

    with open(temp_file, 'w') as fpw, open(file_name, 'r') as fpr:
        for line in fpr:
            data = json.loads(line)
            true_labels.append(data['label'])

            input_data = {
                'id': data['id'],
                'question': 'Has the patient had recent events?',
                'passage': data['pred_str'].split('[/INST]')[-1]
            }
            json.dump(input_data, fpw)
            fpw.write('\n')

    cmd = f'''python seqclass.py \
        --cache_dir CACHE \
        --task boolq \
        --do_predict \
        --model_name_or_path {model_path} \
        --pred_file {temp_file} \
        --predict_dir {class_model_output_dir} \
        --output_dir {class_model_output_dir} \
        --save_steps 10000 \
        --max_seq_length 240'''
    
    subprocess.run(cmd, shell=True)

    with open(f'{class_model_output_dir}/predictions.tsv', 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter='\t')
        data = list(reader)

    clear_output(wait=True)

    predicted_labels = [int(x[1]) for x in data[1:]]
    correct = sum(x == y for x, y in zip(true_labels, predicted_labels))

    print(f"Correct: {correct}\t\t Accuracy: {round(correct / len(true_labels), 2)}")
    display_f1(true_labels, predicted_labels)
    display_conf_matrix_fig(true_labels, predicted_labels, name, fig_name)


    subprocess.run(cmd, shell=True)
    clear_output(wait=True)

    with open(f'{class_model_output_dir}/predictions.tsv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        predicted_labels = [int(row[1]) for row in list(reader)[1:]]

    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    print(f"Correct: {correct}\t\t Accuracy: {round(correct / len(true_labels), 2)}")
    display_f1(true_labels, predicted_labels)
    display_conf_matrix_fig(true_labels, predicted_labels, name, fig_name)

def run_evaluation(file_name, name, fig_name, mode='regex', class_model_output_dir=None, labels_file=None, temp_file = '/temp.txt', model_path = None):
    """
    Dispatch evaluation based on mode.
    mode: 'regex' or 'cot'
    """
    if mode == 'regex':
        eval_regex(file_name, name, fig_name, labels_file)
    elif mode == 'cot':
        if not class_model_output_dir:
            raise ValueError("class_model_output_dir must be specified for COT mode.")
        eval_cot(file_name, name, class_model_output_dir, fig_name, temp_file, model_path)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'regex' or 'cot'.")

def comparison_plot(file_name, models, accuracies):
    plt.scatter(models, accuracies, color='blue', marker='o')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracies')
    plt.tight_layout()
    plt.show()
    plt.gcf().savefig(f'figures/{file_name}', dpi=100)


import argparse

def main():
    parser = argparse.ArgumentParser(description="Run classification evaluation using either regex or COT model.")
    
    parser.add_argument("--file_name", type=str, required=True, help="Path to the prediction file (JSONL).")
    parser.add_argument("--name", type=str, required=True, help="Label for the evaluation (used in plots/titles).")
    parser.add_argument("--fig_name", type=str, default=None, help="Filename to save confusion matrix plot (optional).")
    parser.add_argument("--mode", type=str, choices=["regex", "cot"], required=True, help="Classification mode: 'regex' or 'cot'.")
    parser.add_argument("--class_model_output_dir", type=str, default=None, help="Output dir for COT model predictions.")
    parser.add_argument("--labels_file", type=str, default=None, help="Optional ground-truth label file (for regex mode).")
    parser.add_argument("--temp_file", type=str, default='/temp.txt',
                    help="Temporary file path for intermediate JSONL (for COT mode).")
    parser.add_argument("--model_path", type=str, default = None,
                    help="Path to the classification model directory (for COT mode).")


    args = parser.parse_args()

    run_evaluation(
        file_name=args.file_name,
        name=args.name,
        fig_name=args.fig_name,
        mode=args.mode,
        class_model_output_dir=args.class_model_output_dir,
        labels_file=args.labels_file,
        temp_file = args.temp_file,
        model_path = args.model_path,
    )

if __name__ == "__main__":
    main()
