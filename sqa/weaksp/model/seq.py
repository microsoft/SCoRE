import torch
import logging
from torch import nn
import random

import torch.nn.functional as F
from typing import Union, List, Dict, Any, Set
from collections import defaultdict

from ..sempar.action_walker import ActionSpaceWalker
from ..sempar.context.table_question_context import TableQuestionContext
from ..sempar.domain_languages.wikitable_abstract_language import WikiTableAbstractLanguage
from ..model.baseline import Programmer
from ..module.linked_seq2seq import LinkedSeq2Seq

from allennlp.semparse.domain_languages import ParsingError, ExecutionError
from allennlp.semparse.domain_languages.domain_language import FunctionType, BasicType

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def log_sum_exp(score_list: List):
    if isinstance(score_list, list):
        score_v = torch.stack(score_list, 0)
    else:
        score_v = score_list
    ret_v = score_v - F.log_softmax(score_v, dim=0)
    ret_scalar = ret_v.mean(0)
    return ret_scalar

class SeqProgrammer(Programmer):
    """
    Sketch is generated by a seq2seq model
    """
    def __init__(self, *args):
        super(SeqProgrammer, self).__init__(*args)

        # score for candidate
        if self.use_roberta:
            self.col2action = nn.Linear(self.var_token_embed_size + self.column_type_embed_size +
                self.column_indicator_size, self.sketch_prod_embed_size)
            self.row2action = nn.Linear(self.sketch_prod_embed_size + self.var_token_embed_size +
                self.column_type_embed_size + self.column_indicator_size + self.rnn_hidden_size * 2,
                self.sketch_prod_embed_size)

            self.col2feat = nn.Linear(self.var_token_embed_size + self.column_type_embed_size +
                self.column_indicator_size, self.slot_hidden_score_size)
            self.row2feat = nn.Linear(self.sketch_prod_embed_size + self.var_token_embed_size +
                self.column_type_embed_size + self.column_indicator_size + self.rnn_hidden_size * 2,
                self.slot_hidden_score_size)
        else:
            self.col2action = nn.Linear(self.var_token_embed_size + self.column_type_embed_size +
                self.column_indicator_size, self.sketch_prod_embed_size)
            self.row2action = nn.Linear(self.sketch_prod_embed_size + self.var_token_embed_size +
                self.column_type_embed_size + self.column_indicator_size + self.rnn_hidden_size * 2,
                self.sketch_prod_embed_size)

            self.col2feat = nn.Linear(self.var_token_embed_size + self.column_type_embed_size +
                self.column_indicator_size, self.slot_hidden_score_size)
            self.row2feat = nn.Linear(self.sketch_prod_embed_size + self.var_token_embed_size +
                self.column_type_embed_size + self.column_indicator_size + self.rnn_hidden_size * 2,
                self.slot_hidden_score_size)
        
        self.col_feat2score = nn.Bilinear(self.slot_hidden_score_size, self.rnn_hidden_size * 4, 1)
        self.row_feat2score = nn.Bilinear(self.slot_hidden_score_size, self.rnn_hidden_size * 4, 1)


        # seq2seq
        self.seq2seq = LinkedSeq2Seq(self.rnn_hidden_size * 2, self.sketch_prod_embed,
                                self.sketch_prod2id, self.rnn_hidden_size * 2,
                                self.col2action, self.row2action,
                                self.col2feat, self.row2feat,
                                self.col_feat2score, self.row_feat2score,
                                self.device)

    def forward(self,
            context: TableQuestionContext,
            sketch2program: Dict) -> torch.Tensor:
        world = WikiTableAbstractLanguage(context)

        # encode questions
        token_in_table_feat = context.question_in_table_feat
        token_encodes, token_reps = self.encode_question(context.question_tokens, token_in_table_feat)

        sketch_lf2actions = self.sketch_lf2actions(world)
        consistent_scores = []
        candidate_rep_dic = self.construct_candidates(world, token_encodes)
        for sketch_lf in sketch2program:
            sketch_actions = sketch_lf2actions[sketch_lf]
            if len(sketch2program[sketch_lf]) > self.CONSISTENT_INST_NUM_BOUND:
                continue
            for program_lf in sketch2program[sketch_lf]:
                try:
                    program_actions = world.logical_form_to_action_sequence(program_lf)
                except:
                    # logger.info(f"Program: {program_lf}")
                    # very hacky to support select with multiple argument/return type
                    str_type = BasicType("str")
                    cache = world._function_types["select"][:]
                    world._function_types["select"] = [FunctionType([
                        world._function_types["previous"][:][0].argument_types[0],
                        world._function_types["mode"][:][0].argument_types[1]],
                        world._function_types["mode"][:][0].return_type) ]
                    program_actions = world.logical_form_to_action_sequence(program_lf)
                    world._function_types["select"] = cache
                seq_log_likeli = self.seq2seq(world, token_reps, token_encodes,
                    candidate_rep_dic, sketch_actions, program_actions)
                if seq_log_likeli:
                    consistent_scores.append(seq_log_likeli)

        if len(consistent_scores) > 0:
            return -1 * log_sum_exp(consistent_scores)
        else:
            return None


    def evaluate(self,
            context: TableQuestionContext,
            sketch2program: Dict) -> bool:
        world = WikiTableAbstractLanguage(context)
        ret_dic = defaultdict(int)

        # encode question and offline sketches
        token_in_table_feat = context.question_in_table_feat
        token_encodes, token_reps = self.encode_question(context.question_tokens, token_in_table_feat)
        candidate_rep_dic = self.construct_candidates(world, token_encodes)

        sketch_actions, program_actions = self.seq2seq.decode(world, token_reps, \
                token_encodes, candidate_rep_dic)

        sketch_lf = world.action_sequence_to_logical_form(sketch_actions)
        program_lf = world.action_sequence_to_logical_form(program_actions)
        if sketch_lf in sketch2program:
            sketch_triggered = True
            if program_lf in sketch2program[sketch_lf]:
                lf_triggered = True
            else:
                lf_triggered = False
        else:
            sketch_triggered = False
            lf_triggered = False

        ret_dic["best_program_lf"] = program_lf
        ret_dic["best_program_actions"] = program_actions
        ret_dic["best_sketch_lf"] = sketch_lf
        ret_dic["best_sketch_actions"] = sketch_actions

        return ret_dic
