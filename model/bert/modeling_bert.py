import math
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import torch.nn.functional as F


"""
Loading the config file here
"""
class BertConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from None


def load_bert_config(config_path = None):
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / "Config"
            / "bert_config.yaml"
        )
    
    with open(config_path ,encoding="utf-8") as config_file:
        return BertConfig(yaml.safe_load(config_file))


class BertEmbedding(nn.Module):
    def __init__(self , config):
        super().__init__()
        #print(config['vocab_size'])
        self.word_embedding = nn.Embedding(
            config.vocab_size , config.hidden_size
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings , config.hidden_size
        )
        self.segment_embedding = nn.Embedding(
             config.no_of_seq , config.hidden_size
        )


        self.LayerNorm = nn.LayerNorm(config.hidden_size , eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self , input_ids: torch.Tensor , no_of_seq: torch.Tensor = None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand_as(input_ids)
        
        if no_of_seq is None:
            no_of_seq = torch.zeros_like(input_ids)

        embeddings = (
            self.word_embedding(input_ids)
            + self.position_embedding(position_ids)
            + self.segment_embedding(no_of_seq)
        )

        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)
    

class BertSelfAttention(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size// config.num_attention_heads

        self.query =nn.Linear(config.hidden_size , config.hidden_size)
        self.key = nn.Linear(config.hidden_size , config.hidden_size)
        self.value = nn.Linear(config.hidden_size , config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def _split_head(self , x:torch.Tensor, batch_size:int):
        x = x.view(batch_size , -1, self.num_heads , self.head_dim)
        return x.permute(0 , 2 , 1, 3) #batch , head_number , seq _len , head dimnsion size
    
    def forward(self , hidden_states:torch.Tensor , attention_mask : torch.Tensor = None):
        batch_size = hidden_states.size(0)

        q = self._split_head(self.query(hidden_states) , batch_size)
        k = self._split_head(self.key(hidden_states) , batch_size)
        v = self._split_head(self.value(hidden_states) , batch_size)

        scores = torch.matmul(q , k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        probs = F.softmax(scores , dim = -1)
        probs = self.dropout(probs)

        context = torch.matmul(probs , v)# batch , no_of_head , seq_len , head_dimension
        
        #return to original dimension here
        context = context.permute(0, 2, 1, 3)
        """
        using reshape because the context is uncontiguous in nature .
        The result is non-contiguous, and .view() requires a compatible memory layout,
        so it raises the stride error.
        """
        return context.reshape(batch_size, -1, self.num_heads * self.head_dim)


class BertSelfOutput(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size , config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size , eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self , hidden_states: torch.Tensor , input_tensor:torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        """And here we will add the residual block too"""
        return self.LayerNorm(hidden_states + input_tensor)

"""
The main full attention layer here
"""
class BertFullAttention(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self, hidden_state: torch.Tensor):
        self_output = self.attention(hidden_state)
        return self.output(self_output , hidden_state )

"""
bert feedforward layer here
"""
class BertFeedForward(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size , config.intermediate_size)
        self.activation_fn = F.gelu[config.hidden_size]
        self.dense_2 = nn.Linear(config.intermediate_size , config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size , eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self , hidden_state: torch.Tensor , input_tensor: torch.Tensor):
        self.output_1 = self.dense_1(hidden_state)
        self.output_1 = self.activation_fn(self.output_1)
        self.output_2 = self.dense_2(self.output_1)
        self.output_2 = self.dropout(self.output_2)
        """
        Use of skip connection here
        """
        return self.LayerNorm(self.output_2 + input_tensor)


"""
Main Bert one block architecture here
"""
class BertLayer(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.attention = BertFullAttention(config)
        self.feed_forward = BertFeedForward(config)
    
    def forward(self , hidden_state: torch.Tensor):
        attn_output = self.attention(hidden_state)
        intermediate_output = self.feed_forward(attn_output)
        return intermediate_output

"""
Multiple attention block 
here we are repeating teh BertLayer multiple time here
"""
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self , hidden_states: torch.Tensor):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

"""
The full Bert model 
"""
class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = BertEmbedding(config)
        self.encoder = BertEncoder(config)
    
    def forward(self , input_ids:torch.Tensor = None , no_of_seq:torch.Tensor = None):
        embedding_output = self.embedding(input_ids , no_of_seq)
        sequence_output = self.encoder(embedding_output)




    


