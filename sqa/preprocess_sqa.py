import argparse
import os
import json
import copy
import pickle
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict, Counter

import logging
logger = logging.getLogger("root")  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

from allennlp.data.dataset_readers.semantic_parsing.wikitables import util as wikitables_util
from allennlp.data.tokenizers.token import Token
from allennlp.semparse.domain_languages import ParsingError, ExecutionError

from weaksp.reader.util import load_jsonl, load_jsonl_table
from weaksp.reader.reader import WTReader

from weaksp.sempar.action_walker import ActionSpaceWalker
from weaksp.sempar.context.table_question_context import TableQuestionContext
from weaksp.sempar.domain_languages.wikitable_language import WikiTablesLanguage
from weaksp.sempar.domain_languages.wikitable_abstract_language import WikiTableAbstractLanguage

def prepare_corenlp(sqa_path, table_file, corenlp_input_output_list, corenlp_input_output):
    table_dict = load_jsonl_table(table_file)

    sqa_data = {}

    train_examples = []
    with open(os.path.join(sqa_path, 'train.tsv')) as infile:
        for line in infile:
            train_examples.append(line.strip().split('\t'))

    test_examples = []
    with open(os.path.join(sqa_path, 'test.tsv')) as infile:
        for line in infile:
            test_examples.append(line.strip().split('\t'))

    examples = train_examples[1:] + test_examples[1:]
    print('Number of SQA examples:', len(examples))

    table_id_list = []
    for example in examples:
        qid = example[0]
        annotator = example[1]
        position = example[2]
        question_id = f"{qid}_ann{annotator}_pos{position}"
        question = example[3]
        table_id = 't_'+example[4].split('/')[1].split('.')[0]
        table_id_list.append(table_id)
        table_lines = table_dict[table_id]["raw_lines"]
        tokenized_question = [Token(token) for token in question.split()]
        table_context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)

        answer_cell = eval(example[5])
        answer_cell = sorted([eval(cell) for cell in answer_cell])
        
        answer = []
        try:
            for cell in answer_cell:
                row_index = cell[0]
                column_index = cell[1]
                column_name = table_context.column_index_to_name[column_index]
                column_name = f"string_column:{column_name}"
                cell_value = table_context.table_data[row_index][column_name]
                answer.append(cell_value)
        except:
            print(question_id, "cannot extract answers by answer cell; using answer string")
            answer = eval(example[6].strip('"').replace('""','"')) # a list

        if qid not in sqa_data:
            sqa_data[qid] = {}
        if annotator not in sqa_data[qid]:
            sqa_data[qid][annotator] = []
        sqa_data[qid][annotator].append((position, question, answer))
    
    print('Number of Unique Tables:', len(set(table_id_list)))
    print('Creating inputs for CoreNLP...', end='')
    f_list = open(corenlp_input_output_list, 'w')
    for qid in sqa_data:
        for annotator in sqa_data[qid]:
            sqa_data[qid][annotator] = sorted(sqa_data[qid][annotator], key=lambda x: x[0]) # sort by position
            for position in range(len(sqa_data[qid][annotator])):
                questions_position = [q[1] for q in sqa_data[qid][annotator][:position+1]]
                questions_position.reverse()
                question_merge = ' <s> '.join(questions_position)
                fname = os.path.join(os.path.abspath(corenlp_input_output), '{}_ann{}_pos{}_question.txt'.format(qid, annotator, position))
                f_list.write(fname+'\n')
                with open(fname, 'w') as f:
                    f.write(question_merge+'\n')

                answer = sqa_data[qid][annotator][position][2]
                fname = os.path.join(os.path.abspath(corenlp_input_output), '{}_ann{}_pos{}_answer.txt'.format(qid, annotator, position))
                with open(fname, 'w') as f:
                    f.write('\n'.join(answer)+'\n')
    f_list.close()
    print('Done')
    return sqa_data


def create_raw_input_split(qid2tableid, split, corenlp_input_output, split_tagged, fraction=None):
    # read split
    sqa_data = {}

    examples = []
    with open(split) as infile:
        cnt = 0
        for line in infile:
            cnt += 1
            if cnt == 1:
                continue
            examples.append(line.strip().split('\t'))
    for example in examples:
        qid = example[0]
        annotator = example[1]
        position = example[2]
        if qid not in sqa_data:
            sqa_data[qid] = {}
        if annotator not in sqa_data[qid]:
            sqa_data[qid][annotator] = {}
        sqa_data[qid][annotator][position] = {}

    qid_list = list(sqa_data.keys())
    if fraction:
       import random
       random.shuffle(qid_list)
       fraction_num = int(fraction*len(qid_list))
       qid_list = qid_list[:fraction_num]


    for qid in qid_list:
        for annotator in sqa_data[qid]:
            for position in sqa_data[qid][annotator]:
                # read corenlp_input_output
                corenlp_input = os.path.join(corenlp_input_output, '{}_ann{}_pos{}_question.txt'.format(qid, annotator, position))
                corenlp_output = os.path.join(corenlp_input_output, '{}_ann{}_pos{}_question.txt.json'.format(qid, annotator, position))

                with open(corenlp_input) as infile:
                    questions = infile.read().strip()
                with open(corenlp_output) as infile:
                    data = json.load(infile)
                tokens = []
                lemmaTokens = []
                posTags = []
                nerTags = []
                nerValues = []
                for sentence in data['sentences']:
                    for token in sentence['tokens']:
                        tokens.append(token['word'])
                        lemmaTokens.append(token['lemma'])
                        posTags.append(token['pos'])
                        nerTags.append(token['ner'])
                        if 'normalizedNER' in token:
                            nerValues.append(token['normalizedNER'])
                        else:
                            nerValues.append('')

                corenlp_input = os.path.join(corenlp_input_output, '{}_ann{}_pos{}_answer.txt'.format(qid, annotator, position))
                with open(corenlp_input) as infile:
                    answer_list = [line.strip() for line in infile.readlines()]

                sqa_data[qid][annotator][position]['utterance'] = questions
                sqa_data[qid][annotator][position]['context'] = qid2tableid[qid]
                sqa_data[qid][annotator][position]['targetValue'] = '|'.join(answer_list)
                sqa_data[qid][annotator][position]['tokens'] = '|'.join(tokens)
                sqa_data[qid][annotator][position]['lemmaTokens'] = '|'.join(lemmaTokens)
                sqa_data[qid][annotator][position]['posTags'] = '|'.join(posTags)
                sqa_data[qid][annotator][position]['nerTags'] = '|'.join(nerTags)
                sqa_data[qid][annotator][position]['nerValues'] = '|'.join(nerValues)
                sqa_data[qid][annotator][position]['targetCanon'] = sqa_data[qid][annotator][position]['targetValue']
                sqa_data[qid][annotator][position]['targetCanonType'] = 'undefined'

    # output
    with open(split_tagged, 'w') as f:
        f.write('\t'.join(['id', 'utterance', 'context', 'targetValue', 'tokens',
                          'lemmaTokens', 'posTags', 'nerTags', 'nerValues', 'targetCanon', 'targetCanonType']))
        f.write('\n')
        for qid in qid_list:
            for annotator in sqa_data[qid]:
                for position in sqa_data[qid][annotator]:
                    f.write('\t'.join(['{}_ann{}_pos{}'.format(qid, annotator, position),
                                      sqa_data[qid][annotator][position]['utterance'],
                                      sqa_data[qid][annotator][position]['context'],
                                      sqa_data[qid][annotator][position]['targetValue'],
                                      sqa_data[qid][annotator][position]['tokens'],
                                      sqa_data[qid][annotator][position]['lemmaTokens'],
                                      sqa_data[qid][annotator][position]['posTags'],
                                      sqa_data[qid][annotator][position]['nerTags'],
                                      sqa_data[qid][annotator][position]['nerValues'],
                                      sqa_data[qid][annotator][position]['targetCanon'],
                                      sqa_data[qid][annotator][position]['targetCanonType']]))
                    f.write('\n')


def prepare_raw_input(sqa_path, corenlp_input_output, raw_input):
    examples = []

    with open(os.path.join(sqa_path, 'train.tsv')) as infile:
        cnt = 0
        for line in infile:
            cnt += 1
            if cnt == 1:
                continue
            examples.append(line.strip().split('\t'))

    with open(os.path.join(sqa_path, 'test.tsv')) as infile:
        cnt = 0
        for line in infile:
            cnt += 1
            if cnt == 1:
                continue
            examples.append(line.strip().split('\t'))
    
    qid2tableid = {}
    for example in examples:
        qid = example[0]
        annotator = example[1]
        position = example[2]
        table_id = example[4].split('/')[1].split('.')[0].split('_')
        qid2tableid[qid] = 'csv/{}-csv/{}.csv'.format(table_id[0],table_id[1])

    train_split = os.path.join(sqa_path, 'random-split-1-train.tsv')
    dev_split = os.path.join(sqa_path, 'random-split-1-dev.tsv')
    test_split = os.path.join(sqa_path, 'test.tsv')

    if not os.path.exists(os.path.join(raw_input, 'data')):
        os.makedirs(os.path.join(raw_input, 'data'))

    train_split_tagged = os.path.join(raw_input, 'data/training.tagged')
    dev_split_tagged = os.path.join(raw_input, 'data/dev.tagged')
    test_split_tagged = os.path.join(raw_input, 'data/test.tagged')

    create_raw_input_split(qid2tableid, train_split, corenlp_input_output, train_split_tagged)
    create_raw_input_split(qid2tableid, dev_split, corenlp_input_output, dev_split_tagged)
    create_raw_input_split(qid2tableid, test_split, corenlp_input_output, test_split_tagged)

    return


def cache_data(table_file, train_file, dev_file, test_file, embed_file, output_file):
    tables = load_jsonl_table(table_file)
    train_examples = load_jsonl(train_file)
    dev_examples = load_jsonl(dev_file)
    test_examples = load_jsonl(test_file)

    wt_reader = WTReader(tables, train_examples, dev_examples, test_examples, embed_file)
    wt_reader.gen_vocab()
    wt_reader.gen_glove()
    wt_reader.check()
    with open(output_file, "wb") as f:
        pickle.dump(wt_reader, f)

    return


def coverage_example(example: Dict, table_lines: Dict, max_sketch_length: int) -> int :
    question_id = example["id"]
    output_file_pointer = open(os.path.join('data/processed_sqa/mp/', question_id), "w")

    utterance = example["question"]

    sketch_candidates = [] 
    table_id = example["context"]

    target_value, target_can = example["answer"] # (targeValue, targetCan)

    tokens = []
    assert len(example["tokens"]) == len(example["processed_tokens"])
    for t,p_t in zip(example["tokens"], example["processed_tokens"]):
        if t in ['<START>', '<DECODE>']:
            tokens.append(p_t)
        else:
            tokens.append(t)
    example["tokens"] = tokens
    tokenized_question = [ Token(token) for token in  example["tokens"]]

    context = TableQuestionContext.read_from_lines(table_lines, tokenized_question)
    context.take_corenlp_entities(example["entities"])
    world = WikiTableAbstractLanguage(context)
    walker = ActionSpaceWalker(world)

    print(f"{question_id} {utterance}", file=output_file_pointer)
    print(f"Table: {table_id}", file=output_file_pointer)
    sketch2lf = defaultdict(list)
    all_logical_forms = walker.get_logical_forms_by_sketches(max_sketch_length, None)
    # output the correct logical form
    for sketch, logical_form in all_logical_forms:
        sketch = world.action_sequence_to_logical_form(sketch)
        if world.evaluate_logical_form(logical_form, target_value, target_can):
            sketch2lf[sketch].append(logical_form)

    question_id = example["id"]
    utterance = example["question"]
    if len(sketch2lf) == 0:
        print("NO LOGICAL FORMS FOUND!", file=output_file_pointer)
        coverage_counter = 0
    else:
        coverage_counter = 1
    for sketch in sketch2lf:
        print("Sketch:", sketch, file=output_file_pointer)
        for lf in sketch2lf[sketch]:
            print("\t", lf, file=output_file_pointer)

    print(file=output_file_pointer)
    print(file=output_file_pointer)

    output_file_pointer.close()

    return coverage_counter


def coverage(examples: Dict,
           max_sketch_length: int, 
           table_dict: Dict,
           output_path: str) -> None :
    coverage_counter = 0
    examples_filter = []
    table_lines_filter = []
    for example in examples:
        table_id = example["context"]
        examples_filter.append(example)
        table_lines_filter.append(table_dict[table_id]["raw_lines"])

    pool = mp.Pool(processes=10)
    results = [pool.apply_async(coverage_example, args=(example, table_lines, max_sketch_length,)) for example, table_lines in zip(examples_filter, table_lines_filter)]
    output = [p.get() for p in results]

    coverage_counter = sum(output)

    for example in examples_filter:
        question_id = example["id"]
        output_file_pointer = os.path.join('data/processed_sqa/mp/', question_id)
        command = f"cat {output_file_pointer} >> {output_path}"
        os.system(command)

    print(f"Coverage: {coverage_counter}/{len(examples)}")

    return


def search_program(exp_id, max_sketch_length, table_file, train_file, dev_file, test_file):
    print(f"Exp id: {exp_id}")

    # load examples
    train_examples = load_jsonl(train_file)
    dev_examples = load_jsonl(dev_file)
    test_examples = load_jsonl(test_file)
    tables = load_jsonl_table(table_file)

    train_examples = train_examples + dev_examples

    wt_reader = WTReader(tables, train_examples, [], test_examples, None)
    wt_reader.check()

    # evaluate the sketches
    output_path = f"data/processed_sqa/{exp_id}.train.programs"
    coverage(wt_reader.train_examples, max_sketch_length, wt_reader.table_dict, output_path)

    output_path = f"data/processed_sqa/{exp_id}.test.programs"
    coverage(wt_reader.test_examples, max_sketch_length, wt_reader.table_dict, output_path)

    return


def cache_program(program_file_name, section, output_filename):
    _dc_sketch_list = []
    id2question = [] 
    question2program = dict()
    program_counter = 0
    coverage = dict()
    with open(program_file_name, "r") as f:
        for line in f:
            line = line[:-1]
            if section == "test":
                flag = ["nu-"]
            else:
                flag = ["nt-", "ns-"]
            # if line.startswith(flag):
            if line[:3] in flag:
                q_id = line.split()[0]
                line = next(f)[:-1]
                lh, rh = line.split()
                assert lh == "Table:"
                q_t_pair = (q_id, rh)
                id2question.append(q_t_pair)

                qid, annotator, position = q_id.split('_')
                annotator = annotator[3:]
                position = position[3:]
                # TODO
                q_t_pair = (f"{qid}_ann{annotator}_pos{position}", rh)
                if qid not in coverage:
                    coverage[qid] = {}
                if annotator not in coverage[qid]:
                    coverage[qid][annotator] = {}
                coverage[qid][annotator][position] = 0

                line = next(f)[:-1]
                if line == "NO LOGICAL FORMS FOUND!":
                    line = next(f) #blank line
                    continue
                coverage[qid][annotator][position] = 1

                _sketch2program = defaultdict(list)
                while line:
                    assert line.startswith("Sketch: ")
                    _dc_sketch_list.append(line)
                    sketch = line.strip()[8:]

                    line = next(f)[:-1] 
                    while line.startswith("\t"):
                        program_counter += 1
                        line = line.strip()
                        _sketch2program[sketch].append(line)
                        line = next(f)[:-1] 
                
                question2program[q_t_pair] = _sketch2program

    print(f"Raw number of sketch (from program): \
        {len(set(_dc_sketch_list))}")
    print(f"Raw number of programs: {program_counter}")

    # double check
    counter = 0
    with open(program_file_name, "r") as f:
        for line in f:
            if line.startswith("\t"):
                counter += 1
    print(f"double check # of programs: {counter}")

    with open(output_filename, 'wb') as f:
        pickle.dump(question2program, f)

    # print coverage stats
    all_cnt = 0
    all_cnt_correct = 0
    seq_cnt = 0
    seq_cnt_correct = 0
    q1_cnt = 0
    q1_cnt_correct = 0
    q2_cnt = 0
    q2_cnt_correct = 0
    q3_cnt = 0
    q3_cnt_correct = 0

    last_cnt_correct = 0

    f = open('fail_list.txt', 'w')

    for qid in coverage:
        for annotator in coverage[qid]:
            seq_cnt += 1
            seq_correct = True
            for position in coverage[qid][annotator]:
                all_cnt += 1
                result_i = coverage[qid][annotator][position]
                if result_i == 0:
                    f.write(f"{qid}_ann{annotator}_pos{position}"+'\n')
                all_cnt_correct += result_i
                if result_i == 0:
                    seq_correct = False
                if position == '0':
                    q1_cnt += 1
                    q1_cnt_correct += result_i
                if position == '1':
                    q2_cnt += 1
                    q2_cnt_correct += result_i
                if position == '2':
                    q3_cnt += 1
                    q3_cnt_correct += result_i
                if int(position) == len(coverage[qid][annotator])-1:
                    last_cnt_correct += result_i
            if seq_correct:
                seq_cnt_correct += 1
    f.close()
    all_accuracy = 100 * float(all_cnt_correct) / all_cnt
    seq_accuracy = 100 * float(seq_cnt_correct) / seq_cnt
    q1_accuracy = 100 * float(q1_cnt_correct) / q1_cnt
    q2_accuracy = 100 * float(q2_cnt_correct) / q2_cnt
    q3_accuracy = 100 * float(q3_cnt_correct) / q3_cnt
    print('ALL: ', all_cnt_correct, all_cnt)
    print('SEQ: ', seq_cnt_correct, seq_cnt)
    print('Q1: ', q1_cnt_correct, q1_cnt)
    print('Q2: ', q2_cnt_correct, q2_cnt)
    print('Q3: ', q3_cnt_correct, q3_cnt)
    print('Last: ', last_cnt_correct)
    print('ALL SEQ Q1 Q2 Q3')
    print("{:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(all_accuracy, seq_accuracy, q1_accuracy, q2_accuracy, q3_accuracy))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SQA")
    parser.add_argument("--step", choices=["prepare_corenlp", "prepare_raw_input", "cache_data", "search_program", "cache_program"], help="step in preprocessing")
    # prepare_corenlp
    parser.add_argument("--sqa_path", help="")
    parser.add_argument("--table_file", help="")
    parser.add_argument("--corenlp_input_output_list", help="")
    parser.add_argument("--corenlp_input_output", help="")
    # prepare_raw_input
    parser.add_argument("--raw_input", help="")
    # cache_data
    parser.add_argument("--train_file", help="")
    parser.add_argument("--dev_file", help="")
    parser.add_argument("--test_file", help="")
    parser.add_argument("--embed_file", help="")
    parser.add_argument("--output_file", help="")
    # search_program
    parser.add_argument("--exp_id", help="")
    parser.add_argument("--max_sketch_length", type=int, help="")
    # cache_program
    parser.add_argument("--program_file_name", help="")
    parser.add_argument("--section", help="")
    parser.add_argument("--output_filename", help="")

    args = parser.parse_args()

    if args.step == "prepare_corenlp":
        if not os.path.exists(args.corenlp_input_output):
            os.makedirs(args.corenlp_input_output)
        prepare_corenlp(args.sqa_path, args.table_file, args.corenlp_input_output_list, args.corenlp_input_output)
    elif args.step == "prepare_raw_input":
        prepare_raw_input(args.sqa_path, args.corenlp_input_output, args.raw_input)
    elif args.step == "cache_data":
        cache_data(args.table_file, args.train_file, args.dev_file, args.test_file, args.embed_file, args.output_file)
    elif args.step == "search_program":
        log_path = "log/eval_coverage_debug.log"
        if not os.path.exists('log/'):
            os.makedirs('log/')
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        if not os.path.exists('data/processed_sqa/mp'):
            os.makedirs('data/processed_sqa/mp')
        search_program(args.exp_id, args.max_sketch_length, args.table_file, args.train_file, args.dev_file, args.test_file)
    elif args.step == "cache_program":
        cache_program(args.program_file_name, args.section, args.output_filename)
