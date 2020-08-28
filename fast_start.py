import torch
from transformers import *

MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased'),
          (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model, GPT2Tokenizer, 'gpt2'),
          (CTRLModel, CTRLTokenizer, 'ctrl'),
          (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
          (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
          (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
          (RobertaModel, RobertaTokenizer, 'roberta-base'),
          (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
          ]

model_class, tokenizer_class, pretrained_weights = BertModel, BertTokenizer, 'bert-base-uncased'

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Encode text
input_ids = torch.tensor([tokenizer.encode("Here is some text to encode",
                                           add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
