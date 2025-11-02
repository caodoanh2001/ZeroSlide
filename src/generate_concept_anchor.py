import os
import torch
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
from prompts import brca_prompts, nsclc_prompts, rcc_prompts, cesc_prompts, esca_prompts, tgct_prompts
from transformers import AutoModel

save_path = "./prompt_feats/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

dict_concepts = {
    'brca': brca_prompts(),
    'nsclc': nsclc_prompts(),
    'rcc': rcc_prompts(),
    'cesc': cesc_prompts(),
    'esca': esca_prompts(),
    'tgct': tgct_prompts()
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conch_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
conch_model = conch_model.to(device)

conch_model.eval()
tokenizer = get_tokenizer()

for dataset in dict_concepts:
    cls_templates = dict_concepts[dataset]
    feats = []
    for i in range(len(cls_templates)):
        tokenized_templates = tokenize(texts=cls_templates[i], tokenizer=tokenizer).to(device)
        feats.append(conch_model.encode_text(tokenized_templates).detach())

    torch.save(feats, os.path.join(save_path, dataset + "_concepts.pt"))