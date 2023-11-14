from torch import nn
import torch
import torch.nn.functional as F
import os
from modules.imagebind.models.imagebind_model import ImageBindModel
import configs.config_baseline as default_config
from modules.confede.decoder.classifier import BaseClassifier
from modules.imagebind.models.imagebind_model import ModalityType
from modules.imagebind.data import load_and_transform_text as tokenize

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class Baseline(nn.Module):
    def __init__(self,
                 encoder_fea_dim=1024,
                 drop_out=0.5,
                 config=default_config):
        super(Baseline, self).__init__()

        self.imagebind = ImageBindModel(
            vision_embed_dim=1280,
            vision_num_blocks=32,
            vision_num_heads=16,
            text_embed_dim=1024,
            text_num_blocks=24,
            text_num_heads=16,
            out_embed_dim=1024,
            audio_drop_path=0.1,
            imu_drop_path=0.7,
        )

        hidden_size = [encoder_fea_dim, int(encoder_fea_dim / 2), int(encoder_fea_dim / 4), int(encoder_fea_dim / 8), ]

        self.config = config
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
        self.num_classes = len(EMOTION_LABELS)

        self.decoder = BaseClassifier(input_size=encoder_fea_dim * 3,
                                      hidden_size=hidden_size,
                                      output_size=self.num_classes, drop_out=drop_out,
                                      name='NaiveClassifier', )

        self.criterion = nn.MSELoss()

        self.device = config.DEVICE

    def encode(self, text, vision, video_mask, audio):
        batch_size, fcnt, c, h, w = vision.shape
        vision = vision.view(batch_size * fcnt, c, h, w)
        inputs = {
            ModalityType.TEXT: text,
            ModalityType.VISION: vision,
            ModalityType.AUDIO: audio
        }
        embeddings = self.imagebind(inputs)
        v_embed = embeddings[ModalityType.VISION]
        a_embed = embeddings[ModalityType.AUDIO]
        t_embed = embeddings[ModalityType.TEXT]
        v_embed = v_embed.view(batch_size, fcnt, -1)
        mean_embeds = []
        for i in range(batch_size):
            # perform mean pooling to get an 'average' representation for each video clip
            mean_embed = torch.mean(v_embed[i, :int(torch.sum(video_mask[i]))], dim=0)
            mean_embeds.append(mean_embed)
        return t_embed, torch.stack(mean_embeds, dim=0), a_embed

    def forward(self,
                sample,
                return_loss=True):
        # read modality features and labels
        text = tokenize(sample["text"], device=self.device)
        vision = sample["vision"].clone().detach().to(self.device)
        audio = sample["audio"].clone().detach().to(self.device)
        label = [self.class_to_idx[class_name] for class_name in sample["emotion"]]
        label = F.one_hot(torch.tensor(label), self.num_classes).to(self.device)

        # encode sample with imagebind
        x_t_embed, x_v_embed, x_a_embed = self.encode(text, vision, sample["video_mask"], audio)

        # simple fusion
        x_all = torch.cat((x_t_embed, x_v_embed, x_a_embed), dim=-1)

        # make multimodal prediction
        pred = self.decoder(x_all)
        pred_result = torch.argmax(pred, dim=-1)
        pred_loss = self.criterion(pred.float(), label.float())

        if return_loss:
            return pred_result, pred_loss
        else:
            return pred_result, None

    def freeze_imagebind(self):
        for name, param in self.imagebind.named_parameters():
            if 'adapters' not in name:
                param.requires_grad = False

    def load_model(self, load_pretrain=False, load_checkpoint_epoch=None):
        if load_pretrain:
            encoder_path = os.path.join(self.config.MELD.Path.checkpoints_path, 'imagebind_huge.pth')
            self.imagebind.load_state_dict(torch.load(encoder_path), strict=False)
        else:
            checkpoint_path = os.path.join(self.config.MELD.Path.checkpoints_path,
                                           'baseline_' + str(load_checkpoint_epoch) + '.pth')
            self.load_state_dict(torch.load(checkpoint_path))

    def save_model(self, epoch=None):
        model_path = os.path.join(self.config.MELD.Path.checkpoints_path, 'baseline_' + str(epoch) + '.pth')
        print("Model saved at {}".format(model_path))
        torch.save(self.state_dict(), model_path)
