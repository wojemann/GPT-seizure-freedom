
'''Train or run a model on a sequence classification task - either BoolQA or NLI
Takes inspiration from httpsdiscuss.huggingface.cotquestion-answering-bot-yes-no-answers4496
and from httpsgithub.comhuggingfacetransformersblobmasterexamplespytorchquestion-answeringrun_qa.py'''

import json
import argparse
import numpy as np
from scipy.special import softmax
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed

#Input arguments
parser = argparse.ArgumentParser(description='Trains a model on a classification dataset')
parser.add_argument(--task, help='The name of the task, either NLI or BoolQ',
                    required=True)
parser.add_argument(--model_name_or_path, help='The name of the dataset in huggingface hubs, or the path to its folder',
                    required=True)
parser.add_argument(--dataset_name, help=The name to the dataset in the huggingface hub. Exclusive with --train_file, --validation_file, and --pred_file,
                    required=False)
parser.add_argument(--train_file, help=The path to the training set (.json),
                    required=False)
parser.add_argument('--validation_file', help=The path to the validation set (.json), 
                    required=False)
parser.add_argument('--pred_file', help=The path to the prediction (.json), 
                    required=False)
parser.add_argument('--cache_dir', help=The cache dir, required=True)
parser.add_argument('--save_steps', help='The number of steps before a checkpoint is saved', required=True, type=int)
parser.add_argument('--max_seq_length', help=The maximum tokenized length of the model, required=True, type=int)
parser.add_argument('--output_dir', help='Where to save the trained model or its predictions', required=True)
parser.add_argument('--num_labels', help='The number of classification labels',required=False, type=int, default=3)
parser.add_argument('--do_train', help='Run training', required=False, action='store_true')
parser.add_argument('--do_eval', help=Run evaluation, required=False, action='store_true')
parser.add_argument('--do_predict', help='Make predictions without labels', required=False, action='store_true')
parser.add_argument('--seed', help='What seed do you want', required=False, default=42, type=int)
parser.add_argument('--learning_rate', help='What learning rate do you want', required=False, default=3e-5, type=float)
parser.add_argument('--num_train_epochs', help='How many training epochs do you want', required=False, default=4, type=int)
parser.add_argument('--batch_size', help='Batch Size', required=False, default=4, type=int)
args = parser.parse_args()

#set the seed
set_seed(args.seed)

#check that the task is either NLI or BoolQ
if not (args.task.lower() == 'nli' or args.task.lower() == 'boolq')
    raise ValueError(f'Incorrect task specified. Please choose either NLI or BoolQ.nYou put {args.task}')

#tokenizer for the dataset
def tokenize_function(examples)
    if args.task.lower() == 'nli'
        return tokenizer(examples['hypothesis'], 
                         examples['premise'], 
                         max_length=args.max_seq_length, 
                         stride=128, 
                         truncation=only_second,
                         padding='max_length')
    elif args.task.lower() == 'boolq'
        return tokenizer(examples['question'], 
                         examples['passage'], 
                         max_length=args.max_seq_length, 
                         stride=128, 
                         truncation=only_second,
                         padding='max_length')
    else
        raise ValueError(f'Incorrect task specified. Please choose either NLI or BoolQ.nYou put {args.task}')

#performance metrics
metric_acc = load_metric(accuracy)
def compute_metrics(eval_pred)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels)

if args.dataset_name
    raw_dataset = load_dataset(args.dataset_name,
                               cache_dir=args.cache_dir)
    
    #check that the dataset does not have any -1 values (for SNLI, the -1 value corresponds to a no-answer
    #if -1 values exist, then remove them
    for split in raw_dataset
        raw_dataset[split] = raw_dataset[split].select([i for i,x in enumerate(raw_dataset[split]['label']) if x != -1])
else
    #prepare the raw dataset
    data_files = {}
    if args.do_train
        data_files['train']=args.train_file
    if args.do_eval
        data_files['validation']=args.validation_file
    if args.do_predict
        data_files['test']=args.pred_file
    if len(data_files)  1
        raise ValueError(No training, validation or prediction specified)
    raw_dataset = load_dataset('json', 
                               data_files=data_files,
                               cache_dir=args.cache_dir)
    
print(Dataset Loaded, Preparing Model)

#load in a tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, num_labels=args.num_labels)

#tokenize the dataset
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, load_from_cache_file = False)
print(Tokenized dataset)

#training arguments
train_args = TrainingArguments(output_dir=args.output_dir,
                               per_device_train_batch_size=args.batch_size,
                               learning_rate=args.learning_rate,
                               num_train_epochs=args.num_train_epochs,
                               save_steps=args.save_steps
                               )

print('Models prepared, beginning training')

#prepare the trainer
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_dataset['train'] if args.do_train else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

if args.do_train
    #train the model
    training_results = trainer.train()
    trainer.save_model()
    trainer.save_state()
    trainer.log_metrics('train', training_results.metrics)
    trainer.save_metrics('train', training_results.metrics)

if args.do_eval
    #make predictions and evaluate on the validation set
    validation_results = trainer.predict(tokenized_dataset['validation'])
    trainer.log_metrics(eval, validation_results.metrics)
    trainer.save_metrics(eval, validation_results.metrics)
    #save predictions to file
    with open(args.output_dir+eval_predictions.tsv, 'w') as f
        f.write('PredictionstTrue_Labeltargmaxtmax_softmaxtID')
        f.write('n')
        for idx in range(len(validation_results.predictions))
            f.write(str(validation_results.predictions[idx]))
            f.write('t')
            f.write(str(validation_results.label_ids[idx]))
            f.write('t')
            f.write(str(np.argmax(validation_results.predictions[idx])))
            f.write('t')
            f.write(str(np.max(softmax(validation_results.predictions[idx]))))
            f.write('t')
            f.write(str(tokenized_dataset['validation']['id'][idx]))
            f.write('n')

if args.do_predict
    #make predictions on the prediction set
    prediction_results = trainer.predict(tokenized_dataset['test'])
    print(Saving predictions in +args.output_dir+predictions.tsv)
    with open(args.output_dir+predictions.tsv, 'w') as f
        f.write('logitstargmaxtmax_softmaxtID')
        f.write('n')
        for idx in range(len(prediction_results.predictions))
            f.write(str(prediction_results.predictions[idx]))
            f.write('t')
            f.write(str(np.argmax(prediction_results.predictions[idx])))
            f.write('t')
            f.write(str(np.max(softmax(prediction_results.predictions[idx]))))
            f.write('t')
            f.write(str(tokenized_dataset['test']['id'][idx]))
            f.write('n')

