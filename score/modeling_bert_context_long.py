import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertLayerNorm, BertPreTrainedModel, gelu, BertModel

COLUMN_SQL_LABEL_COUNT = 502
SQL_DIFF_LABEL_COUNT = 120

class BertForContext(BertPreTrainedModel):
    
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.lm_head = BertContextHead(config)
        self.q_tab_dense = nn.Linear(config.hidden_size*2, config.hidden_size)

        self.init_weights()

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None,
        masked_col_labels=None,
        masked_context_labels=None,
        q_tab_inds=None,
        is_train=True
    ):
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        sequence_output = outputs[0]
        
        if q_tab_inds is not None:
            q_tab_inds = q_tab_inds.unsqueeze(2).expand_as(sequence_output)
            q_tab_output = torch.gather(sequence_output, 1, q_tab_inds)
            sequence_output = self.q_tab_dense(torch.cat([sequence_output, q_tab_output], 2))
            
        lm_prediction_scores, col_prediction_scores, context_prediction_scores = self.lm_head(sequence_output)
        
        total_loss = None
        if masked_col_labels is not None:
            # TODO: weights for labels
            weight_list = [0.3] + [1.]*(COLUMN_SQL_LABEL_COUNT-1)
            weights = torch.tensor(weight_list).cuda()
            weighted_loss_fct = CrossEntropyLoss(weight=weights, ignore_index=-1)
            masked_col_loss = weighted_loss_fct(col_prediction_scores.view(-1, COLUMN_SQL_LABEL_COUNT), masked_col_labels.view(-1))
            
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            
        if masked_col_labels is not None and masked_lm_labels is not None:
            total_loss = 0.8 * masked_col_loss + 0.2 * masked_lm_loss
        elif masked_col_labels is not None:
            total_loss = masked_col_loss
        elif masked_lm_labels is not None:
            total_loss = masked_lm_loss
                    
        if is_train:
            return total_loss
        else:
            return total_loss, (lm_prediction_scores, col_prediction_scores, context_prediction_scores)
        # return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertContextHead(nn.Module):
    """Bert Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.dense_col = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm_col = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder_col = nn.Linear(config.hidden_size, COLUMN_SQL_LABEL_COUNT, bias=False)
        self.bias_col = nn.Parameter(torch.zeros(COLUMN_SQL_LABEL_COUNT))
        
        self.dense_context = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm_context = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder_context = nn.Linear(config.hidden_size, SQL_DIFF_LABEL_COUNT, bias=False)
        self.bias_context = nn.Parameter(torch.zeros(SQL_DIFF_LABEL_COUNT))

    def forward(self, features, **kwargs):
        lm_prediction_scores = self.dense(features)
        lm_prediction_scores = gelu(lm_prediction_scores)
        lm_prediction_scores = self.layer_norm(lm_prediction_scores)
        # project back to size of vocabulary with bias
        lm_prediction_scores = self.decoder(lm_prediction_scores) + self.bias

        col_prediction_scores = self.dense_col(features)
        col_prediction_scores = gelu(col_prediction_scores)
        col_prediction_scores = self.layer_norm_col(col_prediction_scores)
        # project back to size of possible column labels
        col_prediction_scores = self.decoder_col(col_prediction_scores) + self.bias_col
        
        context_prediction_scores = self.dense_context(features)
        context_prediction_scores = gelu(context_prediction_scores)
        context_prediction_scores = self.layer_norm_context(context_prediction_scores)
        # project back to size of possible sql diff labels
        context_prediction_scores = self.decoder_context(context_prediction_scores) + self.bias_context

        return lm_prediction_scores, col_prediction_scores, context_prediction_scores
