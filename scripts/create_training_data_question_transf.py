import codecs
import spacy
import math
import random
from collections import defaultdict
#from qanom.candidate_extraction.prepare_qanom_prompts import get_verb_forms_from_lexical_resources
import random
# sentence role predicate (verbal infinitive)   : question (two options: role name or role description...)
# QASRL / QANOM
nlp = spacy.load("en_core_web_sm")

import pandas as pd
import re

def load_frames(frame_path):
    frames = pd.read_csv(frame_path, sep="\t")
    frames = frames[frames.sense_particle == "_"].copy()
    frames = frames[['predicate_lemma', 'role_type', 'role_desc']].drop_duplicates()
    frames = frames.groupby(['predicate_lemma', 'role_type']).head(5)
    frames = frames.groupby(['predicate_lemma', 'role_type']).role_desc.apply(" ; ".join).reset_index()
    frames.sort_values(['predicate_lemma', 'role_type'], inplace=True)
    frames = frames.to_dict(orient='records')
    frame_to_desc = {(f['predicate_lemma'], f['role_type']): f['role_desc']
                     for f in frames}
    return frame_to_desc


def create_data_other_format():
    question_file = '/Users/valentinapyatkin/PycharmProjects/CrossSRL/Data/QASRL/qasrl_qanom.filled.dev.tsv'

    # ID pred arg : ok
    source_file = codecs.open('question_transformation/val.source', 'w')
    target_file = codecs.open('question_transformation/val.target', 'w')

    for file_numb, file in enumerate([question_file]):
        input_file = pd.read_csv(file, sep='\t')
        for index, row in input_file.iterrows():
            if isinstance(row['filled_question'], str):
                sentence = row['text'].split()
                pred = row['predicate_lemma']
                proto_question = row['proto_question']
                filled_questions = row['filled_question'].split('~!~')
                filled_question = random.choice(filled_questions)
                predicate_index = int(row['predicate_span'].split(':')[0])
                marked_sentence = []
                for token_idx, token in enumerate(sentence):
                    if token_idx == predicate_index:
                        marked_sentence.append('PREDICATE_START')
                        marked_sentence.append(token)
                        marked_sentence.append('PREDICATE_END')
                    else:
                        marked_sentence.append(token)
                doc = nlp(pred)
                predicate_lemma = ''
                for token in doc:
                    predicate_lemma = token.lemma_
                # sentence_predicate_marked proto_question predicate_lemma
                # julian_question
                source_file.write(' '.join(marked_sentence) + ' </s> ' + predicate_lemma + ' [SEP] ' + proto_question +'\n')
                print(filled_question)
                target_file.write(filled_question + '\n')
    source_file.close()
    target_file.close()


create_data_other_format()