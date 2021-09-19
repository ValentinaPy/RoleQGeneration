# from transformers import BartForConditionalGeneration, BartTokenizer
#
#
#
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", force_bos_token_to_be_generated=True)
# tok = BartTokenizer.from_pretrained("facebook/bart-large")
# # Original question: How isn't the boys being taught ?
# example_english_phrase = "The boys aren't playing in the garden."
# question_phrase = "Where <mask> the boys playing?"
# batch = tok(example_english_phrase, question_phrase, return_tensors='pt')
# generated_ids = model.generate(batch['input_ids'])
# print(tok.batch_decode(generated_ids, skip_special_tokens=True))
import csv
import sys

from transformers import AutoModelWithLMHead, AutoTokenizer, BertForMaskedLM, pipeline
import torch

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
# model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
# # isn't
# sequence = f"Where {tokenizer.mask_token} particles dissolved?"
# input = tokenizer.encode(sequence, return_tensors="pt")
# mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
#
# token_logits = model(input)[0]
# mask_token_logits = token_logits[0, mask_token_index, :]
#
# top_5_tokens = torch.topk(mask_token_logits, 60, dim=1).indices[0].tolist()
#
# for token in top_5_tokens:
#     print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

def correct_julian_qasrl():
    model = pipeline("fill-mask", model='bert-base-cased', device=0, top_k=100)
    infile = csv.DictReader(open('/home/nlp/pyatkiv/workspace/CrossSRL/Data/QASRL/qasrl_qanom.filled.train.tsv'), delimiter='\t')
    header = list(infile.fieldnames)
    header.append('fixed_question')
    outfile = csv.DictWriter(open('/home/nlp/pyatkiv/workspace/CrossSRL/Data/QASRL/qasrl_qanom.filled.corrected.train.tsv', 'w'), delimiter='\t', fieldnames=header)
    outfile.writeheader()
    pairs_dict = {'does': 'do', 'do': 'does', 'is': 'are', 'are': 'is', 'has':'have', 'have':'has', 'was':'were', 'were':'was'}
    negation_dict = {"doesn't": "don't", "don't": "doesn't", "isn't": "aren't", "aren't": "isn't", "hasn't":"haven't", "haven't":"hasn't", "wasn't":"weren't", "weren't":"wasn't"}
    neg_non_neg = {"doesn't": "does", "don't": "do", "isn't": "is", "aren't": "are", "hasn't":"has", "haven't":"have", "wasn't":"was", "weren't":"were"}
    triggers = ["does", "do", "is", "are", "has", "have", "was", "were", "doesn't", "don't", "isn't", "aren't", "hasn't", "haven't", "wasn't", "weren't"]

    for row in infile:
        questions = row["filled_question"].split('~!~')
        fixed_questions = []
        for question in questions:
            question = question.split()
            found = False
            for word_idx, word in enumerate(question):
                if word in triggers:
                    found = True
                    opposite_negation = ''
                    if word in pairs_dict:
                        target_word = word
                        opposite_word = pairs_dict[word]
                    elif word in neg_non_neg:
                        target_word = neg_non_neg[word]
                        opposite_word = pairs_dict[target_word]
                        opposite_negation = negation_dict[word]
                    sequence = ' '.join(question[:word_idx]+['[MASK]']+question[word_idx+1:])
                    rankings = model(sequence, targets=[target_word, opposite_word])
                    target_prob = 0
                    opposite_prob = 0
                    for rank in rankings:
                        if rank["token_str"] == target_word:
                            target_prob = rank["score"]
                        else:
                            opposite_prob = rank["score"]
                    if opposite_prob > target_prob:
                        if opposite_negation != '':
                            fixed_question = ' '.join(question[:word_idx] + [opposite_negation] + question[word_idx + 1:])
                            print('new')
                            print(question)
                            print(fixed_question)
                        else:
                            fixed_question = ' '.join(question[:word_idx]+[opposite_word]+question[word_idx+1:])
                    else:
                        fixed_question = ' '.join(question)
                    fixed_questions.append(fixed_question)
            if not found:
                fixed_question = ' '.join(question)
                fixed_questions.append(fixed_question)
        out_row = row
        out_row["fixed_question"] = '~!~'.join(fixed_questions)
        outfile.writerow(out_row)

correct_julian_qasrl()