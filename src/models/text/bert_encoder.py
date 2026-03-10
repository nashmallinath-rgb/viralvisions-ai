import torch
import torch.nn as nn
from transformers import BertModel

class ViralBertEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', trainable=False):
        super(ViralBertEncoder, self).__init__()
        # Load the pre-trained BERT weights
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freezing the weights keeps it simple & prevents 
        # the model from 'forgetting' language basics.
        for param in self.bert.parameters():
            param.requires_grad = trainable
            
        # Dropout helps handle the 'noise' of hashtags and slang
        self.dropout = nn.Dropout(0.3)
        self.feature_dim = self.bert.config.hidden_size # 768

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # The 'pooler_output' is the [CLS] token, which is the 
        # mathematical summary of the whole caption.
        pooled_output = outputs.pooler_output
        
        return self.dropout(pooled_output)
