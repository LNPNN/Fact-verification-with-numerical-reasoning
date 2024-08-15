from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

import torch
import sys
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
from urllib.parse import quote
from evidence import ER_BM25_Sbert

import utils

sys.path.append('D:/KLTN/flaskProject')
import config
from utils import evaluate

app = Flask(__name__)

print("Loading model...")
num_labels = 3
# load model XLM-R
pre_trained_xlmr = "xlm-roberta-large"
xlmr_checkpoint_path = "saved_models/xlmr-large/model.pt"

model_class_item_xlmr = utils.MODEL_CLASSES[pre_trained_xlmr]
model_name_xlmr = model_class_item_xlmr['model_name']
do_lower_case_xlmr = model_class_item_xlmr['do_lower_case'] if 'do_lower_case' in model_class_item_xlmr else False
tokenizer_xlmr = model_class_item_xlmr['tokenizer'].from_pretrained(model_name_xlmr,
                                                          cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                          do_lower_case=do_lower_case_xlmr)

model_xlmr = model_class_item_xlmr['sequence_classification'].from_pretrained(model_name_xlmr,
                                                                    cache_dir=str(
                                                                        config.PRO_ROOT / "trans_cache"),
                                                                    num_labels=num_labels)

model_xlmr.load_state_dict(torch.load(xlmr_checkpoint_path, map_location=torch.device('cpu')))

# load model PhơBERT

pre_trained_phobert = "phobert-large"
phobert_checkpoint_path = "saved_models/phobert-large/model.pt"

model_class_item_phobert = utils.MODEL_CLASSES[pre_trained_phobert]
model_name_phobert = model_class_item_phobert['model_name']
do_lower_case_phobert = model_class_item_phobert['do_lower_case'] if 'do_lower_case' in model_class_item_phobert else False
tokenizer_phobert = model_class_item_phobert['tokenizer'].from_pretrained(model_name_phobert,
                                                          cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                          do_lower_case=do_lower_case_phobert)

model_phobert = model_class_item_phobert['sequence_classification'].from_pretrained(model_name_phobert,
                                                                    cache_dir=str(
                                                                        config.PRO_ROOT / "trans_cache"),
                                                                    num_labels=num_labels)

model_phobert.load_state_dict(torch.load(phobert_checkpoint_path, map_location=torch.device('cpu')))

# load model infoxlm nhưng là infoxlm

pre_trained_infoxlm = "infoxlm-large"
infoxlm_checkpoint_path = "saved_models/infoxlm/model.pt"

model_class_item_infoxlm = utils.MODEL_CLASSES[pre_trained_infoxlm]
model_name_infoxlm = model_class_item_infoxlm['model_name']
do_lower_case_infoxlm = model_class_item_infoxlm['do_lower_case'] if 'do_lower_case' in model_class_item_infoxlm else False
tokenizer_infoxlm = model_class_item_infoxlm['tokenizer'].from_pretrained(model_name_infoxlm,
                                                          cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                          do_lower_case=do_lower_case_infoxlm)

model_infoxlm = model_class_item_infoxlm['sequence_classification'].from_pretrained(model_name_infoxlm,
                                                                    cache_dir=str(
                                                                        config.PRO_ROOT / "trans_cache"),
                                                                    num_labels=num_labels)

model_infoxlm.load_state_dict(torch.load(infoxlm_checkpoint_path, map_location=torch.device('cpu')))


# load model cafebert nhưng là cafe

pre_trained_cafebert = "cafebert"
cafebert_checkpoint_path = "saved_models/cafebert/model.pt"

model_class_item_cafebert = utils.MODEL_CLASSES[pre_trained_cafebert]
model_name_cafebert = model_class_item_cafebert['model_name']
do_lower_case_cafebert = model_class_item_cafebert['do_lower_case'] if 'do_lower_case' in model_class_item_cafebert else False
tokenizer_cafebert = model_class_item_cafebert['tokenizer'].from_pretrained(model_name_cafebert,
                                                          cache_dir=str(config.PRO_ROOT / "trans_cache"),
                                                          do_lower_case=do_lower_case_cafebert)

model_cafebert = model_class_item_cafebert['sequence_classification'].from_pretrained(model_name_cafebert,
                                                                    cache_dir=str(
                                                                        config.PRO_ROOT / "trans_cache"),
                                                                    num_labels=num_labels)

model_cafebert.load_state_dict(torch.load(cafebert_checkpoint_path, map_location=torch.device('cpu')))

print("Model loaded!")

def word_segment(sent):
  sent = " ".join(rdrsegmenter.tokenize(sent.replace("\n", " ").lower())[0])
  return sent

@app.route('/', methods=["GET"])
def home():  # put application's code here
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():  # put application's code here
    if request.method == 'POST':
        premise = request.form['premise']
        # if(premise==""):
        #     premise=" "
        hypothesis = request.form['hypothesis']
        # if(hypothesis==""):
        #     hypothesis =" "
        num_evidence = int(request.form['num_evidence'])

        premise_seg = word_segment(premise)
        hypothesis_seg = word_segment(hypothesis)

        per_sup_cafebert,  per_ref_cafebert,  per_nei_cafebert = evaluate(tokenizer_cafebert, model_cafebert, premise, hypothesis)
        per_sup_infoxlm, per_ref_infoxlm, per_nei_infoxlm = evaluate(tokenizer_infoxlm, model_infoxlm,
                                                                                         premise, hypothesis)
        per_sup_phobert, per_ref_phobert, per_nei_phobert = evaluate(tokenizer_phobert, model_phobert,
                                                                                         premise_seg, hypothesis_seg)
        per_sup_xlmr, per_ref_xlmr, per_nei_xlmr = evaluate(tokenizer_xlmr, model_xlmr,
                                                                                         premise, hypothesis)
        
        evidence = ER_BM25_Sbert(premise, hypothesis, num_evidence)
        if per_sup_phobert >= per_ref_phobert and per_sup_phobert >= per_nei_phobert:
            label = "Supported"
        elif per_ref_phobert >= per_sup_phobert and per_ref_phobert >= per_nei_phobert:
            label =  "Refuted"
        else:
            label =  "NotenoughInfo"
            evidence = ["Không có bằng chứng cho nhãn NotenoughInfo"]

        return render_template("index.html", premise_sent=premise, hypothesis_sent=hypothesis, 
                                evidence_sent=evidence, label_sent= label,
                            #    supported_per_cafebert=per_sup_cafebert,
                            #    refuted_pre_cafebert=per_ref_cafebert,
                            #    notenoughinfo_pre_cafebert=per_nei_cafebert,
                            # #    other_pre_cafebert = per_other_cafebert,

                            #    supported_per_infoxlm=per_sup_infoxlm,
                            #    refuted_pre_infoxlm=per_ref_infoxlm,
                            #    notenoughinfo_pre_infoxlm=per_nei_infoxlm,
                            #    # other_pre_infoxlm=per_other_infoxlm,
                            #    #
                            #    supported_per_phobert=per_sup_phobert,
                            #    refuted_pre_phobert=per_ref_phobert,
                            #    notenoughinfo_pre_phobert=per_nei_phobert,
                            #    # other_pre_phobert=per_other_phobert,
                            #    #
                            #    supported_per_xlmr=per_sup_xlmr,
                            #    refuted_pre_xlmr=per_ref_xlmr,
                            #    notenoughinfo_pre_xlmr=per_nei_xlmr,
                            #    # other_pre_xlmr=per_other_xlmr,
                               )
    return render_template("index.html")

if __name__ == '__main__':
    app.run()
