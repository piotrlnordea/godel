# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 21:36:58 2023

@author: latec
"""

# -- coding: utf-8 --
"""
Created on Wed Nov 16 19:00:28 2022

@author: latec
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output
# Instruction for a chitchat task
instruction = f'Instruction: given a dialog context, you need to response empathically.'
instruction = f'Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.'
# Leave the knowldge empty
knowledge = ''' A fresh White House statement has backed the Poland and NATO assessment that the Tuesday Polish border explosion was "most likely the result of a Ukrainian air defense missile that landed in Poland."
The statement by National Security Council Spokesperson Adrienne Watson additionally said the US has "full confidence" in the Polish government's ongoing investigation. "We have seen nothing that contradicts President Duda’s preliminary assessment that this explosion was most likely the result of a Ukrainian air defense missile that unfortunately landed in Poland," she said. 
But like NATO chief Jens Stoltenberg's remarks, she still blamed Russia for the episode. All of these belated admissions that no, it was not a Russian attack, conveniently come well after the US president had seized the 'fog of war' moment yesterday to unveil another massive $37 billion emergency aid package for Ukraine almost simultaneous to the border incident.
A mere less than 24 hours ago, before the dust had settled from the explosion and before investigators could come to any definitive conclusions after the deadly incident on the Polish border village of Przewodów, the Western public was already being harangued and forewarned to stay away from 'conspiracy theories' as the early mainstream headlines - pushed especially based on an anonymous US official in an Associated Press report - were fast out the gate with "Russian missiles hit Poland, killing two". 
'''
dialog = ['who is to blame for Polish border explosion ?',
 ' two Polish men were killed by russian rockets', 'what does A. Watson think ?', 'she blamed Russia for the episode', 'why did she ?', 'probably she does not like russians', ' what Stoltenberg thinks ?', 'probably he does not like russians', 'who is he ?', 'NATO chief Jens Stoltenberg', ' what did Ukraine get ?', 'another massive $37 billion emergency aid package', 'where the accident happen ?', 'in Poland. Polish border village of Przewodów', ' who is behind the bombing ?',
 'who is responsible ?']
print('-------------response-----------')
response = generate(instruction, knowledge, dialog)
print(response)
print('-----------------------')



while True:
    inputString = input('Enter a question:, q to end: ')
    if inputString == "q":
        break
    print('The question str is:', inputString)
    dialog.append(inputString)
    response = generate(instruction, knowledge, dialog)
    print(response)
    dialog.append(response)