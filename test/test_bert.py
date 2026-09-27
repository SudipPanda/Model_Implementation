from pathlib import Path
import sys

# Add parent directory
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))


from model.bert.modeling_bert import BertFullAttention , load_bert_config ,BertModel , BertEmbedding , BertSelfAttention
from transformers import BertTokenizer
import torch
import time

def test_bert_encoder_shape():
    config = load_bert_config()
    embedding = BertEmbedding(config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = "hi it's sudip"
    encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    print("the encode shape is " , encoded["input_ids"].size())
    final_embedding = embedding(encoded["input_ids"], encoded.get("token_type_ids"))
    print("the shape here is " , final_embedding.shape)
    return final_embedding

def test_bert_self_attention(embedding : torch.Tensor):
    start_time = time.process_time()
    attention = BertFullAttention(load_bert_config())
    attention_output = attention(embedding)
    end_time = time.process_time()
    print("total time needed", end_time - start_time )
    print("the attention output shape is", attention_output.shape)
    print("the attention output type is" , type(attention_output))
    
    return attention_output

def test_bert_full_attention(embedding: torch.Tensor):
    start_time = time.process_time()
    attention = BertSelfAttention(load_bert_config())
    attention_output = attention(embedding)
    end_time = time.process_time()
    print("total time needed", end_time - start_time )
    print("the attention output shape is", attention_output.shape)
    print("the attention output type is" , type(attention_output))
    
    return attention_output


if __name__ == "__main__":
    embedding = test_bert_encoder_shape()
    attention_out = test_bert_self_attention(embedding)
    full_attn_out = test_bert_full_attention(embedding)


