from allennlp.predictors.predictor import Predictor
import jsonlines
import codecs
import re
from collections import defaultdict

def read_and_extract_qasrl_pavel_format():
    # sentence : q : a : instance_id : a_start
    outfile = codecs.open('extracted_qasrl_instances_pavel_format.tsv', 'w')
    with jsonlines.open('/Users/vale/PycharmProjects/my-nrl-qasrl/data/qasrl-v2/orig/train.jsonl') as json_file:
        for obj in json_file:
            id = obj["sentenceId"]
            sent = obj["sentenceTokens"]
            for key, value in obj["verbEntries"].items():
                verb_index = [str(key), str(int(key)+1)]
                verb = sent[int(key)]
                stem = value["verbInflectedForms"]["stem"]
                for question, quest_inst in value["questionLabels"].items():
                    answers = quest_inst["answerJudgments"]
                    q_source = quest_inst["questionSources"][0]
                    if "turk" in q_source:
                        ans = []
                        answer_indices = []
                        for ans_instance in answers:
                            if ans_instance["isValid"]:
                                spans = ans_instance['spans']
                                for span in spans:
                                    ans.append(' '.join(sent[span[0]:span[1]]))
                                    answer_indices.append(':'.join([str(s) for s in span]))
                        ans = list(set(ans))
                        outfile.write(' '.join(sent)+'\t'+question+'\t'+'~!~'.join(ans)+'\t'+'~!~'.join(answer_indices)+'\t'+id+'\t'+verb+'\t'+':'.join(verb_index)+'\t'+stem+'\t'+key+'\n')

    outfile.close()

def get_IOU(text1, text2):
    iou = 0
    for answer in text2.split('~!~'):
        t1 = set(text1.lower().split())
        t2 = set(answer.lower().split())
        intersection = t1.intersection(t2)
        union = t1.union(t2)
        if len(intersection) / len(union) > iou:
            iou = len(intersection) / len(union)
    return iou


def check_for_alignment(qa_verb_index, srl_verb_index, srl_arg, qa_answer):
    if qa_verb_index == srl_verb_index:
        arg_iou = get_IOU(srl_arg, qa_answer)
        if arg_iou >= 0.4:
            return True
        else:
            return False

def align_with_qasrl(pred_dict):
    infile = codecs.open('extracted_qasrl_instances_pavel_format.tsv', 'r')
    outfile = codecs.open('good_alignments/withSRL/aligned_qasrl_to_roles_only_args_train.tsv', 'w')
    outfile.write('Sentence\tQASRL_question\tQASRL_answer\tQASRL_answer_indices\tQASRL_id\tQASRL_predicate\tQASRL_predicate_index\tQASRL_predicate_stem\tmodel_target\tmodel_target_token\tmodel_argument\trole\tsrl\n')
    counter = 0
    for inline in infile.readlines():
        line = inline.split('\t')[:-1]
        sent = line[0]
        qa_verb = line[5]
        qa_verb_lemma = line[7]
        qa_question = line[1]
        qa_answers = line[2]
        qa_answer_idx = line[3]
        qa_verb_index = int(line[6].split(':')[0])
        qa_verb_index_orig = line[6]
        ID = line[4]
        found = False
        if ID in pred_dict:
            predictions = pred_dict[ID]
            for prediction in predictions:
                target_verb_idx = int(prediction[-1][0])
                target_verb_word = prediction[-1][1]
                roles_out = [' '.join(role_prediction[0])+' '+role_prediction[1] for role_prediction in prediction[:-1]]
                for arg_inst in prediction[:-1]:
                    arg = ' '.join(arg_inst[0])
                    label = arg_inst[1]
                    alignment = check_for_alignment(qa_verb_index, target_verb_idx, arg, qa_answers)
                    if alignment:
                        outfile.write(
                            sent + '\t' + qa_question + '\t' + qa_answers + '\t' + qa_answer_idx + '\t' + ID + '\t' + qa_verb + '\t' + qa_verb_index_orig + '\t' + qa_verb_lemma + '\t' + str(target_verb_idx) + '\t' + target_verb_word  + '\t' + arg + '\t' + label + '\t'+ '$$$'.join(roles_out)+'\n')
                        print(counter)
                        counter += 1
                        found = True
            if not found:
                print(line)
                print(predictions)
    outfile.close()

def srl_parse_and_align_pavel_format():
    predictor = Predictor.from_path(
          "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz", cuda_device=-1)
    infile = codecs.open('debug.tsv', 'r')
    outfile = codecs.open('bla.tsv', 'w')
    outfile.write('Sentence\tQASRL_question\tQASRL_answer\tQASRL_answer_indices\tQASRL_id\tQASRL_predicate\tQASRL_predicate_index\tQASRL_predicate_stem\tmodel_target\tmodel_target_token\tmodel_argument\trole\n')
    counter = 0
    for inline in infile.readlines():
        line = inline.split('\t')[:-1]
        sent = line[0]
        qa_verb = line[5]
        qa_answers = line[2]
        qa_verb_index = int(line[6].split(':')[0])
        results = predictor.predict(sentence=sent)
        for verb_instance in results["verbs"]:
            srl_verb = verb_instance["verb"]
            arg = []
            labels = []
            all_srl_arguments = []
            all_srl_labels = []
            srl_pred_index = None
            srl_pred = ''
            token_count = 0
            for token, label in zip(results["words"], verb_instance["tags"]):
                if not label.startswith('O') and '-V' not in label:
                    if label.startswith('B'):
                        all_srl_arguments.append(' '.join(arg))
                        all_srl_labels.append(' '.join(labels))
                        arg = []
                        labels = []
                    arg.append(token)
                    labels.append(label)
                if '-V' in label:
                    srl_pred_index = token_count
                    srl_pred = token
                token_count += 1
            all_srl_arguments.append(' '.join(arg))
            all_srl_labels.append(' '.join(labels))
            for srl_arg, labels in zip(all_srl_arguments, all_srl_labels):
                if len(srl_arg.strip())>0:
                    alignment = check_for_alignment(qa_verb_index, srl_pred_index, srl_arg, qa_answers)
                    srl_label = '-'.join(labels.split(' ')[0].split('-')[1:])
                    srl_label = re.sub('ARG', 'A', srl_label)
                    if alignment:
                        outfile.write(inline.strip() + '\t' + srl_verb + '\t' + srl_arg + '\t' + srl_label + '\n')
                        counter += 1
                        print(counter)
                    else:
                        print(qa_verb, srl_pred, srl_arg, ' SEP ',qa_answers, label)
    outfile.close()

def get_predicted():
    #ID : args_senses
    pred_dict = defaultdict(lambda: [])
    infile = jsonlines.open('parse_predictions/qasrl_pb_train_prediction')
    instance_infile = jsonlines.open('/Users/valentinapyatkin/PycharmProjects/QuestionGenerationCrossSRL/data/ForPrediction/qasrl_for_prediction_train.jsonl')
    counter = 0
    for obj, instance in zip(infile, instance_infile):
        print(obj)
        if len(obj["verbs"])>0:
            for verb in obj["verbs"]:
                tags = verb["tags"]
                ID = instance["ID"]
                if tags.count('O')>=len(tags)-1:
                    pass
                else:
                    words = obj["words"]
                    target = []
                    args_senses = []
                    sense = ''
                    arg = []
                    counter = 0
                    for tag, word in zip(tags, words):
                        if tag == 'O':
                            if len(arg)>0:
                                args_senses.append([arg, sense])
                            sense = ''
                            arg = []
                        elif "-V" in tag:
                            if len(arg)>0:
                                args_senses.append([arg, sense])
                            sense = ''
                            arg = []
                            target.append(str(counter))
                            target.append(word)
                        elif tag.startswith('B'):
                            if len(arg)>0:
                                args_senses.append([arg, sense])
                            arg = []
                            sense = '-'.join(tag.split('-')[1:])
                            sense = re.sub('ARG', 'A', sense)
                            arg.append(word)
                        else:
                            new_sense = '-'.join(tag.split('-')[1:])
                            new_sense = re.sub('ARG', 'A', new_sense)
                            if new_sense == sense:
                                arg.append(word)
                        counter += 1
                found = False
                for entry in pred_dict[ID]:
                    if entry[-1] == target:
                        found = True
                if not found and len(target)>0:
                    args_senses.append(target)
                    pred_dict[ID].append(args_senses)
            counter += 1
            #print(counter)
    print('DONE')
    return pred_dict

pred_dict = get_predicted()
align_with_qasrl(pred_dict)
#srl_parse_and_align_pavel_format()
#read_and_extract_qasrl_pavel_format()