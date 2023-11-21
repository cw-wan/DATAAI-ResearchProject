import os
import torch
from torch import nn
import torch.nn.functional as F
from modules.decoder import BaseClassifier
from transformers import AutoModel
from transformers import RobertaTokenizer
from configs import roberta_contrastive_config as base_config
from pytorch_metric_learning.losses import NTXentLoss

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class ContrastiveRoBERTa(nn.Module):
    def __init__(self,
                 config=base_config):
        super(ContrastiveRoBERTa, self).__init__()

        self.config = config
        self.device = self.config.device
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
        self.num_classes = len(EMOTION_LABELS)

        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.Path.roberta, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(self.config.Path.roberta, local_files_only=True)

        self.embedding_size = self.config.Model.embedding_size
        hidden_size = [int(self.embedding_size / 4), ]

        self.decoder = BaseClassifier(input_size=self.embedding_size,
                                      hidden_size=hidden_size,
                                      output_size=self.config.DownStream.output_size)

        self.criterion = nn.BCELoss()
        self.info_nce = NTXentLoss(temperature=self.config.Model.temperature)

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.encoder(input_ids=tokens["input_ids"].to(self.device),
                                  attention_mask=tokens["attention_mask"].to(self.device))["last_hidden_state"]
        return embeddings

    def forward(self, samples, samples2=None, return_loss=True):
        cls = self.encode(samples["text"])[:, 0]
        label = [self.class_to_idx[class_name] for class_name in samples["emotion"]]
        label = F.one_hot(torch.tensor(label), self.num_classes).to(self.device)
        pred = self.decoder(cls)
        if samples2 is not None:
            cls2 = self.encode(samples2["text"])[:, 0]
        if return_loss:
            pred_loss = self.criterion(pred.float(), label.float())
            contrastive_loss = 0
            if samples2 is not None:
                contrastive_labels = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6]).to(self.device)
                for i in range(len(cls)):
                    contrastive_samples = torch.cat((cls[i].unsqueeze(0), cls2[i * 7: (i + 1) * 7]), dim=0)
                    contrastive_loss += self.info_nce(embeddings=contrastive_samples, labels=contrastive_labels)
                contrastive_loss /= len(cls)
            all_loss = pred_loss + contrastive_loss * self.config.Model.contrastive_loss_weight
            return pred, all_loss, pred_loss, contrastive_loss
        else:
            return pred

    def save_model(self, epoch):
        save_path = os.path.join(self.config.Path.save, "roberta_contrastive" + str(epoch) + ".pth")
        print("Contrastive Roberta saved at " + save_path)
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_checkpoint_epoch):
        checkpoint_path = os.path.join(self.config.Path.save,
                                       'roberta_contrastive' + str(load_checkpoint_epoch) + '.pth')
        self.load_state_dict(torch.load(checkpoint_path))
