from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch

MODEL_CLASSES = {
    "cafebert": {
        "model_name": "uitnlp/CafeBERT",
        "tokenizer": AutoTokenizer,
        "sequence_classification": XLMRobertaForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "do_lower_case": True,
    },
    "phobert-large": {
        "model_name": "vinai/phobert-large",
        "tokenizer": AutoTokenizer,
        "sequence_classification": RobertaForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "do_lower_case": True,
    },
    "xlm-roberta-large": {
        "model_name": "xlm-roberta-large",
        "tokenizer": XLMRobertaTokenizer,
        "sequence_classification": XLMRobertaForSequenceClassification,
        # "padding_token_value": 0,
        "padding_segement_value": 0,
        "padding_att_value": 0,
        "do_lower_case": True,
    },
    "infoxlm-large":{
      "model_name": "microsoft/infoxlm-large",
      "tokenizer": AutoTokenizer,
      "sequence_classification": AutoModelForSequenceClassification,
      "padding_segement_value": 0,
      "padding_att_value": 0,
      "do_lower_case": True,
    }
}


def evaluate(tokenizer, model, premise, hypothesis):
    max_length = 256

    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis,
                                                     max_length=max_length,
                                                     return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None)

    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one

    print("Prediction:")
    print("Supported:", round(predicted_probability[0],2))
    print("Refuted:", round(predicted_probability[2],2))
    print("NotenoughInfo:", round(predicted_probability[1],2))
    # print("Other:", round(predicted_probability[3],2))

    return round(predicted_probability[0], 5), round(predicted_probability[2], 5), round(predicted_probability[1], 5)
