from transformers import BertTokenizer, BertModel, AutoModel
from huggingface_hub import snapshot_download

# BERT_MODEL_NAME = "GroNLP/hateBERT"
# model = AutoModel.from_pretrained(BERT_MODEL_NAME)

snapshot_download(repo_id="GroNLP/hateBERT")