# -*- coding: utf-8 -*-
import argparse
import csv
import nltk
import pandas
import os
import re
import time
import json
import pprint

import unicodedata
from codecs import open

from nsm import word_embeddings, data_utils
from table.wtq import evaluator

import tensorflow as tf


# FLAGS
FLAGS = tf.app.flags.FLAGS  

tf.flags.DEFINE_string('raw_input_dir', '',
                       'path.')
tf.flags.DEFINE_string('processed_input_dir', '',
                       'path to the folder to save all the processed data.')
tf.flags.DEFINE_integer('n_train_shard', 90, '.')
tf.flags.DEFINE_integer('max_n_tokens_for_num_prop', 10, '.')
tf.flags.DEFINE_float('min_frac_for_ordered_prop', 0.2, '.')
tf.flags.DEFINE_integer('en_min_tk_count', 10, '.')
tf.flags.DEFINE_bool('use_prop_match_count_feature', False, '.')
tf.flags.DEFINE_bool('anonymize_in_table_tokens', False, '.')
tf.flags.DEFINE_bool('anonymize_datetime_and_number_entities', False, '.')
tf.flags.DEFINE_bool('merge_entities', False, '.')
tf.flags.DEFINE_bool('process_conjunction', False, '.')
tf.flags.DEFINE_bool('expand_entities', False, '.')
tf.flags.DEFINE_bool('use_tokens_contain', False, '.')


# Copied from WikiTableQuestions dataset official evaluator. 
def normalize(x):
    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


# Normalize the date time value from corenlp annotation.
def normalize_date_nervalue(val_string):
    if re.match('\A[\d]{4}$', val_string):
        string = val_string + '-XX-XX'
    elif re.match('\A[\dX]{4}-[\dX]{2}$', val_string):
        string = val_string + '-XX'
    else:
        string = val_string
    val = evaluator.to_value(string)
    if isinstance(val, evaluator.DateValue):
        return val.normalized
    else:
        return None


# Normalize the number value from corenlp annotation.
def normalize_number_nervalue(val_string):
    m = re.search(r'\d+(\.\d+)*', val_string)
    if m is None:
        return None
    string = m.group()
    val = evaluator.to_value(string)
    if isinstance(val, evaluator.NumberValue):
        return val.amount
    else:
        return None

# # Preprocess the tables

# Turn table into KG (implemented as a dictionary).
n_total_num = 0
n_filtered_num = 0
n_date_and_num = 0
n_too_few_num_date = 0


def table2kg(tab_name, data_folder, max_n_tokens_for_num_prop=10, min_frac_for_ordered_prop=0.2):
    kg = {}
    node_info = {}
    tab_ids = tab_name.split('_')
    col_num_to_name = {}
    num_props = set()
    datetime_props = set()
    props = set()
    fn = os.path.join(data_folder, tab_ids[1] + '-tagged', tab_ids[2] + '.tagged')
    
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        # headers are "row col id content tokens lemmaTokens
        # posTags nerTags nerValues number date num2 list
        # listId" in order. 
        header = next(reader)
        # Collect all the column names.
        for row in reader:
            # Assume all the column names are defined in the
            # first row (row index is -1).
            if row[0] != '-1':
                break
            raw_col_name = row[2]
            col_num = row[1]
            # All the column names start with "fb:row.row."
            # like "fb:row.row.title".
            col_name = 'r.' + raw_col_name[11:]
            col_num_to_name[col_num] = col_name
    
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        header = next(reader)
        row_id = ''
        row_node = None
        for row in reader:
            # Ignore all the -1 row.
            if row[0] == '-1':
                continue
            # Create a new row node when a new row starts.
            if row[0] != row_id:
                row_id = row[0]
                row_name = 'row_{}'.format(row_id)
                # Create new row node.
                row_node = dict()
                kg[row_name] = row_node
            col_name = col_num_to_name[row[1]]
            cell_name = 'cell_{}_{}'.format(row[0], row[1])
            node_info[cell_name] = dict(list(zip(header, row)))
            prop_name = col_name + '-string'
            row_node[prop_name] = [normalize(node_info[cell_name]['content'])]
            props.add(prop_name)

            # Number of tokens in this cell.
            n_tokens = len(node_info[cell_name]['tokens'].split('|'))
            if node_info[cell_name]['number']:
                global n_total_num
                n_total_num += 1
            if node_info[cell_name]['number'] and n_tokens >= max_n_tokens_for_num_prop:
                global n_filtered_num
                n_filtered_num += 1
            if node_info[cell_name]['date'] and node_info[cell_name]['number']:
                global n_date_and_num
                n_date_and_num += 1

            # If there are too many tokens in the cell, then
            # it is probably not a number cell, should
            # ignore the numbers extracted by corenlp. 
            if n_tokens < max_n_tokens_for_num_prop:
                num = node_info[cell_name]['number']
                num2 = node_info[cell_name]['num2']
            else:
                num = None
                num2 = None
            date = node_info[cell_name]['date']
            if date:
                prop_name = col_name + '-date'
                row_node[prop_name] = [date]
                datetime_props.add(prop_name)
            if num:
                prop_name = col_name + '-number'
                row_node[prop_name] = [float(num)]
                num_props.add(prop_name)
            if num2:
                prop_name = col_name + '-num2'
                row_node[prop_name] = [float(num2)]
                num_props.add(prop_name)

        row_names = [k for k in list(kg.keys()) if k[:4] == 'row_']
        num_props = list(num_props)
        datetime_props = list(datetime_props)
        
    nodes = list(kg.values())
    total_n = len(nodes)
    
    for prop in (num_props + datetime_props):
        nodes_with_prop = [node for node in nodes if prop in node]
        n = len(nodes_with_prop)
        # If a large fraction of the column don't have
        # number or date, then this column is probably not
        # really number or date.
        if n * 1.0 / total_n < min_frac_for_ordered_prop:
            for node in nodes_with_prop:
                del node[prop]

    return dict(kg=kg, row_ents=row_names,
                props=(list(props) + num_props + datetime_props),
                num_props=num_props, datetime_props=datetime_props)


# Don't use dataframe.from_csv because quotes might be dropped. 
def create_df_from_wtq_questions(fn):
    df_dict = {}
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter='\t', quotechar=None)
        header = next(reader)
        for col in header:
            df_dict[col] = []
        for line in reader:
            for col, val in zip(header, line):
                df_dict[col].append(val)
    df = pandas.DataFrame(df_dict)
    return df


def tokens_contain(string_1, string_2):
  tks_1 = nltk.tokenize.word_tokenize(string_1)
  tks_2 = nltk.tokenize.word_tokenize(string_2)
  return set(tks_2).issubset(set(tks_1))

# preprocess the questions.
def string_in_table_tk(string, kg):
  for k, node in kg['kg'].items():
    for prop, val in node.items():
      if ((isinstance(val[0], str) or isinstance(val[0], bytes)) and
          tokens_contain(val[0], string)):
        return True
      else:
        return False
    
# preprocess the questions.
def string_in_table_str(string, kg):
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if (isinstance(val[0], str) or isinstance(val[0], bytes)) and string in val[0]:
                return True
    else:
        return False

def string_in_table(string, kg):
  if FLAGS.use_tokens_contain:
    return string_in_table_tk(string, kg)
  else:
    return string_in_table_str(string, kg)


def date_in_table(ent, kg):
    props_set = set(kg['datetime_props'])
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop in props_set and ent == val:
                return True
    else:
        return False

        
def num_in_table(ent, kg):
    props_set = set(kg['num_props'])
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop in props_set and ent == val:
                return True
    else:
        return False        


def prop_in_question_score(prop, example, stop_words, binary=True):
    # question = example['question'].decode('utf-8')
    question = example['question']
    if not isinstance(question, str):
        question = question.decode('utf-8')
    prop = prop[2:]
    prop = '-'.join(prop.split('-')[:-1])
    prop_tks = prop.split('_')
    n_in_question = 0
    for tk in prop_tks:
        if not isinstance(tk, str):
            tk = tk.decode('utf-8')
        if tk not in stop_words and tk in question:
            n_in_question += 1
    if binary:
        n_in_question = min(n_in_question, 1)
    return n_in_question


def collect_examples_from_df(df, kg_dict, stop_words):
    examples = []
    for index, row in df.iterrows():
        #print row['utterance'], row['tokens'], index
        match = re.match(r'csv/(?P<first>\d+)-csv/(?P<second>\d+).csv', row['context'])
        context = 't_{}_{}'.format(*match.groups())
        tks = row['tokens'].split('|')
        pos_tags = row['posTags'].split('|')
        vals = row['nerValues'].split('|')
        tags = row['nerTags'].split('|')
        e = dict(id=row['id'], question=row['utterance'],tokens=tks,
                 context=context, pos_tags=pos_tags)
        answer = (row['targetValue'], row['targetCanon'])
        e['answer'] = answer
        e['entities'] = []
        # entities are normalized tokens
        e['processed_tokens'] = tks[:]
        e['in_table'] = [0] * len(tks)
        for i, (tk, tag, val) in enumerate(zip(tks, tags, vals)):
            kg = kg_dict[context]
            if tk not in stop_words:
                if string_in_table(normalize(tk), kg):
                    e['entities'].append(
                        dict(value=[normalize(tk)], token_start=i, token_end=i+1,
                             type='string_list'))
                    e['in_table'][i] = 1
            if tag == 'DATE':
                nerVal = normalize_date_nervalue(val)
                if nerVal is not None:
                    e['entities'].append(
                        dict(value=[nerVal], token_start=i, token_end=i+1,
                             type='datetime_list'))
                    if date_in_table(nerVal, kg):
                        e['in_table'][i] = 1
                        e['processed_tokens'][i] = '<{}>'.format(tag)
                    #e['processed_tokens'][i] += ' <DATE>'
            elif tag == 'NUMBER':
                nerVal = normalize_number_nervalue(val)
                if nerVal is not None:
                    e['entities'].append(
                        dict(value=[nerVal], token_start=i, token_end=i+1,
                             type='num_list'))
                    if num_in_table(nerVal, kg):
                        e['in_table'][i] = 1
                        e['processed_tokens'][i] = '<{}>'.format(tag)
                    #e['processed_tokens'][i] += ' <NUMBER>'
            elif tag != 'O':
                e['processed_tokens'][i] = '<{}>'.format(tag)
        e['features'] = [[it] for it in e['in_table']]
        e['prop_features'] = dict(
          [(prop, [prop_in_question_score(
              prop, e, stop_words,
              binary=not FLAGS.use_prop_match_count_feature)])
           for prop in kg['props']])
        examples.append(e)        
                
    avg_n_ent = (sum([len(e['entities']) for e in examples]) * 1.0 /
                 len(examples))

    print('Average number of entities is {}'.format(avg_n_ent))
    if FLAGS.expand_entities:
      expand_entities(examples, kg_dict)
      avg_n_ent = (sum([len(e['entities']) for e in examples]) * 1.0 /
                   len(examples))
      print('After expanding, average number of entities is {}'.format(avg_n_ent))

    for e in examples:
      e['tmp_tokens'] = e['tokens'][:]

    if FLAGS.anonymize_datetime_and_number_entities:
        for e in examples:
            for ent in e['entities']:
                if ent['type'] == 'datetime_list':
                  for t in range(ent['token_start'], ent['token_end']):
                    e['tmp_tokens'][t] = '<DECODE>'
                elif ent['type'] == 'num_list':
                  for t in range(ent['token_start'], ent['token_end']):
                    e['tmp_tokens'][t] = '<START>'
      
    # if FLAGS.merge_entities:
    #   merge_entities(examples, kg_dict)
    #   avg_n_ent = (sum([len(e['entities']) for e in examples]) * 1.0 /
    #                len(examples))
    #   print 'After merging, average number of entities is {}'.format(avg_n_ent)

    if FLAGS.process_conjunction:
      n_conjunction = process_conjunction(examples, 'or')
      tf.logging.info('{} conjunctions processed.'.format(n_conjunction))
      avg_n_ent = (sum([len(e['entities']) for e in examples]) * 1.0 /
                   len(examples))
      print('After processing conjunction, average number of entities is {}'.format(
        avg_n_ent))

    for e in examples:
      e['tokens'] = e['tmp_tokens']
      
    return examples


# Save the preprocessed examples. 

def dump_examples(examples, fn):
    t1 = time.time()
    with open(fn, 'w') as f:
        for i, e in enumerate(examples):
            f.write(json.dumps(e))
            f.write('\n')
    t2 = time.time()
    print('{} sec used dumping {} examples.'.format(t2 - t1, len(examples)))


def to_unicode(tk, encoding='utf-8'):
  if not isinstance(tk, str):
    return tk.decode(encoding)
  else:
    return tk


def merge_entities(examples, table_dict):
    for e in examples:
        ents = [ent for ent in e['entities']
                if ent['type'] == 'string_list' and ent['value'][0]]
        other_ents = [ent for ent in e['entities'] if ent['type'] != 'string_list']
        kg = table_dict[e['context']]
        l = len(ents)
        new_ents = []
        i = 0
        merged = False
        while i < l:
            top_ent = ents[i].copy()
            new_ents.append(top_ent)
            i += 1
            while i < l:
                if ents[i]['token_start'] - top_ent['token_end'] <= 2:
                    # print e['tokens'][top_ent['token_start']:ents[i]['token_end']]
                    tokens = [to_unicode(tk) for tk in
                              e['tokens'][top_ent['token_start']:ents[i]['token_end']]]
                    ent_tokens = [to_unicode(top_ent['value'][0]),
                                  to_unicode(ents[i]['value'][0])]
                    new_str_1 = ' '.join(tokens)
                    new_str_2 = ' '.join(ent_tokens)
                    new_str_3 = '-'.join(tokens)
                    new_str_4 = '-'.join(ent_tokens)
                    new_str_5 = ''.join(tokens)
                    new_str_6 = ''.join(ent_tokens)
                    # print new_str_1
                    if string_in_table(new_str_1, kg):
                        new_str = new_str_1
                    elif string_in_table(new_str_2, kg):
                        new_str = new_str_2
                    elif string_in_table(new_str_3, kg):
                        new_str = new_str_3
                    elif string_in_table(new_str_4, kg):
                        new_str = new_str_4
                    elif string_in_table(new_str_5, kg):
                        new_str = new_str_5
                    elif string_in_table(new_str_6, kg):
                        new_str = new_str_6
                    else:
                        new_str = ''
                    if new_str:
                        top_ent = dict(value=[new_str], type='string_list', 
                                       token_start=top_ent['token_start'],
                                       token_end=ents[i]['token_end'])
                        new_ents[-1] = top_ent
                        i += 1
                    else:
                        break
                else:
                    break
        e['entities'] = new_ents + other_ents
        for ent in e['entities']:
            for t in range(ent['token_start'], ent['token_end']):
                e['features'][t] = [1]

def expand_entities(examples, table_dict):
    for e in examples:
    # for e in [example_dict['nt-11874']]:
        ents = [ent for ent in e['entities']
                if ent['type'] == 'string_list' and ent['value'][0]]
        other_ents = [ent for ent in e['entities'] if ent['type'] != 'string_list']
        kg = table_dict[e['context']]
        l = len(ents)
        new_ents = []
        i = 0
        tokens = e['tokens']
        for ent in ents:
            # relies on order. 
            if new_ents and ent['token_end'] <= new_ents[-1]['token_end']:
                continue
            else:
                ent['value'][0] = tokens[ent['token_start']]
                new_ents.append(ent)
                while True and ent['token_end'] < len(tokens):
                    new_str_list = (
                      [s.join([to_unicode(ent['value'][0]),
                               to_unicode(tokens[ent['token_end']])]) 
                       for s in [' ', '-', '']] + 
                      [s.join([to_unicode(ent['value'][0]),
                               to_unicode(normalize(tokens[ent['token_end']]))])
                       for s in [' ', '-', '']] + 
                      [s.join([to_unicode(normalize(ent['value'][0])),
                               to_unicode(tokens[ent['token_end']])]) 
                       for s in [' ', '-', '']] + 
                      [s.join([to_unicode(normalize(ent['value'][0])),
                               to_unicode(normalize(tokens[ent['token_end']]))])
                       for s in [' ', '-', '']])
                    for new_str in new_str_list:
                      if string_in_table(new_str, kg):
                        ent['token_end'] += 1
                        ent['value'] = [new_str]
                        break
                    else:
                      break
                ent['value'] = [normalize(ent['value'][0])]
        e['entities'] = new_ents + other_ents


def process_conjunction(examples, conjunction_word, other_words=None):
    i = 0
    for e in examples:
        str_ents = [ent for ent in e['entities'] if ent['type'] == 'string_list']
        other_ents = [ent for ent in e['entities'] if ent['type'] != 'string_list']
        if other_words is not None:
            extra_condition = any([w in e['tokens'] for w in other_words])
        else:
            extra_condition = True
        if str_ents and conjunction_word in e['tokens'] and extra_condition:
            or_idx = e['tokens'].index(conjunction_word)
            before_ent = None
            before_id = None
            after_ent = None
            after_id = None
            for k, ent in enumerate(str_ents):
                if ent['token_end'] <= or_idx:
                    before_ent = ent
                    before_id = k
                    before_distance = abs(ent['token_end'] - or_idx)
                if after_ent is None and ent['token_start'] > or_idx:
                    after_ent = ent
                    after_id = k
                    after_distance = abs(ent['token_start'] - or_idx)
            if (not before_ent is None and not after_ent is None and
                before_distance <= 2 and after_distance <= 2):
                i += 1
                new_ent = dict(
                  value=before_ent['value'] + after_ent['value'],
                  type='string_list',
                  token_start=before_ent['token_start'], 
                  token_end=after_ent['token_end'])
                str_ents[before_id] = new_ent
                del str_ents[after_id]
                e['entities'] = str_ents + other_ents
    return i


def main(unused_argv):
    assert tf.gfile.Exists(FLAGS.raw_input_dir)
    if not tf.gfile.Exists(FLAGS.processed_input_dir):
        tf.gfile.MkDir(FLAGS.processed_input_dir)

    data_folder = FLAGS.raw_input_dir
    table_file = os.path.join(FLAGS.processed_input_dir, 'tables.jsonl')
    test_table_file = os.path.join(FLAGS.processed_input_dir, 'test_table.json')
    stop_words_file = os.path.join(FLAGS.raw_input_dir, 'stop_words.json')

    train_file = os.path.join(FLAGS.processed_input_dir, 'train_examples.jsonl')
    dev_file = os.path.join(FLAGS.processed_input_dir, 'dev_examples.jsonl')
    test_file = os.path.join(FLAGS.processed_input_dir, 'test_examples.jsonl')

    train_tagged = os.path.join(
        FLAGS.raw_input_dir, 'data/training.tagged')
    dev_tagged = os.path.join(
        FLAGS.raw_input_dir, 'data/dev.tagged')
    test_tagged = os.path.join(
        FLAGS.raw_input_dir, 'data/test.tagged')

    # Preprocess the tables.
    subdirs = os.listdir(data_folder)
    subdirs.remove('data')
    subdirs.remove('stop_words.json')

    # Preprocess the tables. 
    table_dict = {}
    folders = []
    t1 = time.time()
    for d in subdirs:
        for fn in os.listdir(os.path.join(data_folder, d)):
            full_path = os.path.join(data_folder, d, fn)
            m = re.match(r'.*/(?P<first>[0-9]*)-tagged/(?P<second>[0-9]*)\.tagged', full_path)
            folders.append(full_path)
            table_name = 't_{}_{}'.format(m.group('first'), m.group('second'))
            kg = table2kg(
                table_name, data_folder,
                max_n_tokens_for_num_prop=FLAGS.max_n_tokens_for_num_prop,
                min_frac_for_ordered_prop=FLAGS.min_frac_for_ordered_prop)
            kg['name'] = table_name
            table_dict[table_name] = kg
    t2 = time.time()
    print(('{} sec used processing the tables.'.format(t2 - t1)))
    print('total number of number cells: {}'.format(n_total_num))
    print('total number of filtered number cells: {}'.format(n_filtered_num))
    print('filtered ration: {}'.format(n_filtered_num * 1.0 / n_total_num))
    print('date and number ratio: {}'.format(n_date_and_num * 1.0 / n_total_num))

    # Save the preprocessed test table. 
    with open(test_table_file, 'w') as f:
        json.dump({'t_203_375': table_dict['t_203_375']}, f)

    # Save the preprocessed table. 
    t1 = time.time()
    with open(table_file, 'w') as f:
        for i, (k, v) in enumerate(table_dict.items()):
            if i % 1000 == 0:
                print('number {}'.format(i))
            f.write(json.dumps(v))
            f.write('\n')
    t2 = time.time()
    print('{} sec used dumping tables'.format(t2 - t1))

    # TODO
    df = create_df_from_wtq_questions(train_tagged)

    with open(stop_words_file, 'r') as f:
        stop_words_list = json.load(f)
    stop_words = set(stop_words_list)
    
    t1 = time.time()
    # TODO
    examples = collect_examples_from_df(
        df, table_dict, stop_words)
    t2 = time.time()
    print('{} sec used collecting train examples.'.format(t2 - t1))
    
    dump_examples(examples, train_file)

    dev_df = create_df_from_wtq_questions(dev_tagged)
    t1 = time.time()    
    dev_examples = collect_examples_from_df(
        dev_df, table_dict, stop_words)
    t2 = time.time()
    print('{} sec used collecting dev examples.'.format(t2 - t1))
    dump_examples(dev_examples, dev_file)

    test_df = create_df_from_wtq_questions(test_tagged)
    t1 = time.time()    
    test_examples = collect_examples_from_df(
        test_df, table_dict, stop_words)
    t2 = time.time()
    print('{} sec used collecting test examples.'.format(t2 - t1))

    dump_examples(test_examples, test_file)


if __name__ == '__main__':  
    tf.app.run()
