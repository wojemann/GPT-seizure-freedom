import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
from tqdm import tqdm

model = None
tokenizer = None
train_dataset = None
date_str = None

def promptize(inputs, task, ft_samples):
    
    if task == 1:  #Zero shot LLAMA Only Yes No. 
        preamble = '''<s>[INST] <<SYS>> You are a board-certified neurologist, your role is to respond with a single word ("Yes," "No," or "Unclear") after carefully reviewing the patient's history. Answer the question: "In the last twelve months, has the patient had epileptic events?" Seizure-free status is determined since the last visit or within the past 12 months, whichever is sooner. Events encompass various types of seizures, auras, myoclonus, staring spells, loss of consciousness, and loss of awareness. Respond only with the designated response words and refrain from providing additional context.<</SYS>> Note: '''
        suffix = ' [/INST]'
    elif task == 1.1: #Zero shot LLAMA reasoning allowed
        preamble = '''<s>[INST] You are a board-certified neurologist, your role is to respond with a single word ("Yes," "No," or "Unclear") after carefully reviewing the patient's history. Answer the question: "In the last twelve months, has the patient had epileptic events?" Seizure-free status is determined since the last visit or within the past 12 months, whichever is sooner. Events encompass various types of seizures, auras, myoclonus, staring spells, loss of consciousness, and loss of awareness. Let's think step by step.<</SYS>> Note:'''
        suffix = ' [/INST]'
    elif task == 1.2: #Zero shot LLAMA COT allowed Prompt Engineering
        preamble = '''<s>[INST] <<SYS>> You are a board-certified neurologist, your role is to respond with a single word ("Yes," "No," or "Unclear") after carefully reviewing the patient's history. Answer the question: "In the last twelve months, has the patient had epileptic events?" Seizure-free status is determined since the last visit or within the past 12 months, whichever is sooner. Events encompass various types of seizures, auras, myoclonus, staring spells, loss of consciousness, and loss of awareness. Seizure-free status is defined as no seizure since the last visit or within the past 12 months, whichever is sooner. Events encompass various seizure types, auras, myoclonus, staring spells, loss of consciousness, feeling off, and loss of awareness. If any of these events occurred within twelve months, the patient has had events; uncertainty is not considered. If the patient has been seizure-free since the last visit, and the last visit was over twelve months ago, the patient is seizure-free. CPS stands for Complex Partial Seizure. Interpret date formats as MM/YY with one dash for month and year or MM/DD/YY for month, day, and year. Surgeries or medication lapses are not events. The note will start with "This note was written in MM/DD/YYYY," and you'll use this date for calculations, considering supporting text. Let's think step by step.. <</SYS>> Note: '''
        suffix = ' [/INST]'
    elif task == 1.4: # date trick
        preamble = f'''<s>[INST] <<SYS>> You are a board-certified neurologist, your role is to respond with a single word ("Yes," "No," or "Unclear") after carefully reviewing the patient's history. Answer the question: "In the last twelve months, has the patient had epileptic events?" Seizure-free status is determined since the last visit or within the past 12 months, whichever is sooner. Events encompass various types of seizures, auras, myoclonus, staring spells, loss of consciousness, and loss of awareness. If any of these events occurred within twelve months, the patient had events; uncertainty is not considered. If the patient has been seizure-free since the last visit, and the last visit was over twelve months ago, the patient is seizure-free. CPS stands for Complex Partial Seizure. Interpret date formats as MM/YY with one dash for month and year or MM/DD/YY for month, day, and year. Surgeries or medication lapses are not events. <</SYS>> [/INST] I am a board-certified epileptologist and will adhere to these instructions. [INST] The note was written on {date_str[1]}/{date_str[2]}/{date_str[0]}.[/INST] Certainly, I'll use the provided date for calculations. Now, please provide relevant information from the patient's medical history regarding epileptic events in the last twelve months. [INST] Let's think step by step. Note: '''
        suffix = ' [/INST]'
    elif task == 1.41: #date trick 1 word
        preamble = f'''<s>[INST] <<SYS>> You are a board-certified neurologist, your role is to respond with a single word ("Yes," "No," or "Unclear") after carefully reviewing the patient's history. Answer the question: "In the last twelve months, has the patient had epileptic events?" Seizure-free status is determined since the last visit or within the past 12 months, whichever is sooner. Events encompass various types of seizures, auras, myoclonus, staring spells, loss of consciousness, and loss of awareness. If any of these events occurred within twelve months, the patient had events; uncertainty is not considered. If the patient has been seizure-free since the last visit, and the last visit was over twelve months ago, the patient is seizure-free. CPS stands for Complex Partial Seizure. Interpret date formats as MM/YY for month and year with one dash or MM/DD/YY for month, day, and year. Surgeries or medication lapses are not events. <</SYS>> [/INST] I am a board-certified epileptologist and will closely follow these instructions. [INST] The note was written on {date_str[1]}/{date_str[2]}/{date_str[0]}. Responses are limited to "Yes" if they have, "No" if they have not, or "Unclear" if uncertain. Provide only one-word responses without additional context. [/INST] Certainly, I'll use the provided date for calculations, responding with one word. [INST] Note:''' 
        suffix = ' [/INST]'

    #Insturction to include date note was written in the reaosning??Look into maybe
    
    elif task == 2.1: # Finetuning with Context (the right way :)
        preamble = f'''<s>[INST]<<SYS>> You are a board-certified neurologist, your role is to respond with a single word ("Yes," "No," or "Unclear") after carefully reviewing the patient's history. Answer the question: "In the last twelve months, has the patient had epileptic events?" Seizure-free status is determined since the last visit or within the past 12 months, whichever is sooner. Events encompass various types of seizures, auras, myoclonus, staring spells, loss of consciousness, and loss of awareness. If any of these events occurred within twelve months, the patient had events; uncertainty is not considered. If the patient has been seizure-free since the last visit, and the last visit was over twelve months ago, the patient is seizure-free. CPS stands for Complex Partial Seizure. Interpret date formats as MM/YY with one dash for month and year or MM/DD/YY for month, day, and year. Surgeries or medication lapses are not events. <</SYS>>'''
        suffix = "Answer: [/INST]"
        for i,sample in train_dataset.iterrows():
            dic = sample.to_dict()
            preamble += "In the last twelve months, has the patient had epileptic events? Note: "
            preamble += dic['passage']
            preamble += " [/INST]Answer: "
            preamble += dic['answer']
            preamble += "</s><s>[INST]"
        preamble += "In the last twelve months, has the patient had epileptic events? Note: "

    elif task == 2.2: # Finetuning with context date trick
        preamble = '''<s>[INST]<<SYS>>You are a board-certified neurologist, your role is to respond with a single word ("Yes," "No," or "Unclear") after carefully reviewing the patient's history. Answer the question: "In the last twelve months, has the patient had epileptic events?" Seizure-free status is determined since the last visit or within the past 12 months, whichever is sooner. Events encompass various types of seizures, auras, myoclonus, staring spells, loss of consciousness, and loss of awareness. If any of these events occurred within twelve months, the patient had events; uncertainty is not considered. If the patient has been seizure-free since the last visit, and the last visit was over twelve months ago, the patient is seizure-free. CPS stands for Complex Partial Seizure. Interpret date formats as MM/YY with one dash for month and year or MM/DD/YY for month, day, and year. Surgeries or medication lapses are not events.<</SYS>>'''
        suffix = "Answer: [/INST]"
        for i,sample in train_dataset.iterrows():
            dic = sample.to_dict()
            preamble += f'''Date:  The note was written on {dic[date_str][1]}/{dic[date_str][2]}/{dic[date_str][0]}.[/INST] Certainly, I'll use the provided date for calculations. Now, please provide the relevant information from the patient's medical history regarding epileptic events within the last twelve months.[INST] In the last twelve months, has the patient had epileptic events? Note:''' 
            preamble += dic['passage']
            preamble += " [/INST]Answer: "
            preamble += dic['answer']
            preamble += "</s><s>[INST]"
        preamble += f'''Date: The note was written on {date_str[1]}/{date_str[2]}/{date_str[0]}.[/INST] Certainly, I'll use the provided date for calculations. Now, please provide the relevant information from the patient's medical history regarding epileptic events within the last twelve months.[INST] In the last twelve months, has the patient had epileptic events? Note: '''
   
    return preamble + inputs + suffix
    #return preamble


        
def load_train_dataset(location):
    global train_dataset
    train_dataset = pd.read_json(location,lines=True)

def set_note_date(pdate_str):
    global date_str
    date_str = pdate_str
            
def load_model(seed):
    token = '' # Place your HF access token here
    set_seed(seed)
    print(f"Seed set to {seed}")
    model_name = "Paramter for model name"
    global model
    global tokenizer 
    
    model = AutoModelForCausalLM.from_pretrained(model_name,token=token,
                                             device_map='auto')
                                             # torch_dtype=torch.float32,
                                             # load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='cache',token=token)   
    print(tokenizer.eos_token)

def predict(passage, task, ft_samples):
    prompt = promptize(passage, task, ft_samples)
    with open('LLAMA/prompts.txt', 'a') as fpre:
            fpre.write(prompt)
            fpre.write('\n\n')
    inputs = tokenizer(prompt,return_tensors='pt')
    inputs = inputs.to('cuda')
    generate_ids = model.generate(inputs.input_ids,max_length= int(4096))
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    response = output.split('[/INST]')[-1]
    return response


def run(read_file, output_file, task, ft_samples):
    dataset = pd.read_json(read_file, lines = True)
    datatype='epi'

    passage_arr = [1,1.1, 1.2,1.4,1.41, 2,2.1, 2.2,2.01, 3.2,3.3, 3.25,3.21, 2.3,5,5.1, 6, 6.1]
    pred_str_arr = []

    output_df = []
    filtered_data = []
    written= False
    for _,sample in tqdm(dataset.iterrows(),total=len(dataset)):
        dic = sample.to_dict()
 
        set_note_date(dic['id'].split('_')[2].split('-')) #extract date
        
        if task in passage_arr:
            pred = predict(dic['passage'], task, ft_samples)
        else:
            pred = predict(dic['pred_str'], task, ft_samples)
            dic['context'] = dic['pred_str']
        with open('LLAMA/predictions.txt', 'a') as fpe:
            fpe.write(pred)
            fpe.write('\n\n')
            
        if not written: 
            print(pred)
            written = True
        dic['prediction'] = pred == 'True'
        dic['pred_str'] = pred
        output_df.append(dic)
    output_df = pd.DataFrame(output_df)
    output_df.to_json(output_file,orient='records',lines=True)
  


def main():
    parser = argparse.ArgumentParser(description="Run LLAMA epilepsy prompt script.")
    parser.add_argument('--read_file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save predictions JSONL')
    parser.add_argument('--task', type=float, required=True, help='Task ID (e.g., 1, 1.1, 2.2)')
    parser.add_argument('--ft_samples', type=str, default='', help='Path to finetuning samples (optional)')
    parser.add_argument('--train_dataset', type=str, help='Optional: training dataset for context prompting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model_name', type=str, required=True, help='Name or path of the model to load')

    args = parser.parse_args()

    global model_name
    model_name = args.model_name

    load_model(args.seed)

    if args.train_dataset:
        load_train_dataset(args.train_dataset)

    run(args.read_file, args.output_file, args.task, args.ft_samples)


if __name__ == '__main__':
    main()