import os
if __name__ == "__main__":
    import sys
    full_path = os.path.realpath(__file__)
    cross_srl_dir = os.path.dirname(os.path.dirname(full_path))
    print("Black magic with python modules!")
    print(cross_srl_dir)
    sys.path.insert(0, cross_srl_dir)

#from flask import Flask, request, redirect
from typing import Tuple, Union, List, Optional, Dict
import spacy
from Demo.role_lexicon.role_lexicon import RoleLexicon
import csv
from collections import defaultdict
from question_translation import QuestionTranslator
#from qanom.candidate_extraction.candidate_extraction import get_verb_forms_from_lexical_resources

transformation_model_path = '/home/nlp/pyatkiv/workspace/transformers/examples/seq2seq/question_transformation_grammar_corrected_who/'
device_number = 0
q_translator = QuestionTranslator.from_pretrained(transformation_model_path, device_id=device_number)
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "tetxtcat","tok2vec", "attribute_ruler"])
lex = RoleLexicon.from_file('files/predicate_roles.ontonotes.tsv')

def read_covered_predicates():
    infile = csv.reader(open('files/covered.tsv'), delimiter='\t')
    #predicate : pos  : mapping
    covered_predicates = defaultdict(lambda: defaultdict(lambda: ''))
    for row in infile:
        covered_predicates[row[0]][row[1]] = row[2]
    return covered_predicates

covered_predicates = read_covered_predicates()

def get_proto_question_dict():
    proto_dict = defaultdict(lambda: '')
    proto_score = defaultdict(lambda: 0)
    with open('../resources/qasrl.prototype_accuracy.ontonotes.tsv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            sense_id = int(row["sense_id"])
            if isinstance(sense_id, int):
                sense_id = f"{sense_id:02d}"
            if row['verb_form']+sense_id+row['role_type'] in proto_dict:
                current_count = proto_score[row['verb_form']+sense_id +row['role_type']]
                considered_count = float(row['squad_f1'])
                if considered_count>current_count:
                    proto_dict[row['verb_form'] +sense_id +row['role_type']] = row['proto_question']
                    proto_score[row['verb_form'] +sense_id +row['role_type']] = considered_count
            else:
                proto_dict[row['verb_form']+ sense_id +row['role_type']]=row['proto_question']
                proto_score[row['verb_form'] + sense_id + row['role_type']] = float(row['squad_f1'])
    return proto_dict

proto_dict = get_proto_question_dict()

class RoleQDemo:
    def __init__(self):
        pass

    def analyze(self, text: str) -> Dict:
        doc = nlp(text)
        tokens = []
        indices = []
        lemmas = []
        pos = []
        for token in doc:
            lemma = token.lemma_
            tokens.append(token.text)
            lemmas.append(lemma)
            pos.append(token.pos_)
            if token.pos_ in ['VERB', 'NOUN'] and lemma in covered_predicates:
                if token.pos_[0].lower() in covered_predicates[lemma]:
                    indices.append(token.i)
        return {"tokens": tokens, "indices": indices, "lemmas": lemmas, "pos": pos}

    def get_rolesets(self, selected_idx: int, lemmas: [], pos: []) -> List:
        selected_lemma = lemmas[selected_idx]
        selected_pos = pos[selected_idx][0].lower()
        all_role_sets = lex.get_all_rolesets(selected_lemma, selected_pos)
        all_role_sets_output = defaultdict(lambda : {})
        for role in all_role_sets:
            all_role_sets_output[role.sense_id]={"roleset_desc": role.role_set_desc, "lemma": role.predicate, "pos": role.pos, "sense_id": role.sense_id}
        return all_role_sets_output.values()

    def get_questions(self, lemma: str, pos: str, sense_id: str, predicate_idx: int, text: str) -> List:
        all_roles = lex.get_roleset(lemma, sense_id, pos)
        print(predicate_idx)
        predicate_span = str(predicate_idx)+':'+str(predicate_idx+1)
        questions_list = []
        samples = []
        protos = []
        for role in all_roles:
            proto_question = proto_dict[lemma+sense_id+role]
            protos.append(proto_question)
            samples.append(
                {'proto_question': proto_question, 'predicate_lemma': lemma,
                 'predicate_span': predicate_span,
                 'text': text})
        contextualized_questions = q_translator.predict(samples)
        for question, role, proto in zip(contextualized_questions, all_roles, protos):
            questions_list.append({"role_type": role, "questions": [{"prototype": proto, "contextualized":question}]})
        print(questions_list)
        return questions_list

def main():
    roleqdemo = RoleQDemo()
    roleqdemo.analyze('John sold a pen to Mary.')
    roleqdemo.get_rolesets(1, ['John', 'sell', 'a', 'pen', 'to', 'Mary'], ['NOUN', 'VERB'])
    roleqdemo.get_questions(['John', 'sell', 'a', 'pen', 'to', 'Mary'], 'sell', 1, 'v', '01')
    roleqdemo.get_questions("sell", "v", "01", 1, "John sold a pen to Mary .")

if __name__ == "__main__":
    main()
