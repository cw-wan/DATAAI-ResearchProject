import os
import torch
from torch import nn
import torch.nn.functional as F
from modules.decoder import BaseClassifier
from transformers import AutoModel
from transformers import RobertaTokenizer
from configs import naive_roberta_config as base_config

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class NaiveRoBERTa(nn.Module):
    def __init__(self,
                 config=base_config):
        super(NaiveRoBERTa, self).__init__()

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

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        embeddings = self.encoder(input_ids=tokens["input_ids"].to(self.device),
                                  attention_mask=tokens["attention_mask"].to(self.device))["last_hidden_state"]
        return embeddings

    def forward(self, samples, return_loss=True):
        cls = self.encode(samples["text"])[:, 0]
        label = [self.class_to_idx[class_name] for class_name in samples["emotion"]]
        label = F.one_hot(torch.tensor(label), self.num_classes).to(self.device)
        pred = self.decoder(cls)
        if return_loss:
            loss = self.criterion(pred.float(), label.float())
            return loss, pred
        else:
            return pred

    def save_model(self, epoch):
        save_path = os.path.join(self.config.Path.save, "naive_roberta_" + str(epoch) + ".pth")
        print("Naive Roberta saved at " + save_path)
        torch.save(self.state_dict(), save_path)

    def load_model(self, load_checkpoint_epoch):
        checkpoint_path = os.path.join(self.config.Path.save,
                                       'naive_roberta_' + str(load_checkpoint_epoch) + '.pth')
        self.load_state_dict(torch.load(checkpoint_path))
