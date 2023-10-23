# Adapted from https://github.com/XpastaX/ConFEDE/blob/Graph_Main/MOSI/model/net/constrastive/TVA_fusion.py
import torch
from modules.confede.decoder.classifier import BaseClassifier
from modules.imagebind.models.imagebind_model import ImageBindModel
from torch import nn
import configs.config_meld as default_config
from modules.imagebind.data import load_and_transform_text as tokenize
from modules.imagebind.models.imagebind_model import ModalityType
import torch.nn.functional as F
from pytorch_metric_learning import losses

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(Projector, self).__init__()

        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            # nn.ReLU(),
            # nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class TVAFusion(nn.Module):
    def __init__(self,
                 encoder_fea_dim=1024,
                 drop_out=0.5,
                 config=default_config):
        super(TVAFusion, self).__init__()

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

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
        self.num_classes = len(EMOTION_LABELS)

        uni_fea_dim = int(encoder_fea_dim / 2)

        self.T_sim_proj = Projector(encoder_fea_dim, uni_fea_dim)
        self.V_sim_proj = Projector(encoder_fea_dim, uni_fea_dim)
        self.A_sim_proj = Projector(encoder_fea_dim, uni_fea_dim)

        self.T_dissim_proj = Projector(encoder_fea_dim, uni_fea_dim)
        self.V_dissim_proj = Projector(encoder_fea_dim, uni_fea_dim)
        self.A_dissim_proj = Projector(encoder_fea_dim, uni_fea_dim)

        hidden_size = [uni_fea_dim * 2, uni_fea_dim, int(uni_fea_dim / 2), int(uni_fea_dim / 4), ]

        self.TVA_decoder = BaseClassifier(input_size=uni_fea_dim * 6,
                                          hidden_size=hidden_size,
                                          output_size=self.num_classes, drop_out=drop_out,
                                          name='TVARegClassifier', )

        self.mono_decoder = BaseClassifier(input_size=uni_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=self.num_classes, drop_out=drop_out,
                                           name='TVAMonoRegClassifier', )

        self.criterion = nn.MSELoss()
        self.heat = config.MELD.Downstream.const_heat
        self.NTXent_loss = losses.NTXentLoss(temperature=self.heat)

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
                sample1,
                sample2=None,
                return_loss=True, ):
        # read modality features and labels
        text1 = tokenize(sample1["text"], device=self.device)
        vision1 = sample1["vision"].clone().detach().to(self.device)
        audio1 = sample1["audio"].clone().detach().to(self.device)
        label1 = [self.class_to_idx[class_name] for class_name in sample1["emotion"]]
        label1 = F.one_hot(torch.tensor(label1), self.num_classes).to(self.device)

        # encode sample 1 with imagebind
        x1_t_embed, x1_v_embed, x1_a_embed = self.encode(text1, vision1, sample1["video_mask"], audio1)

        # perform similarity and dissimilarity decomposition on embeddings of sample 1
        x1_t_sim = self.T_sim_proj(x1_t_embed)
        x1_v_sim = self.V_sim_proj(x1_v_embed)
        x1_a_sim = self.A_sim_proj(x1_a_embed)
        x1_t_dissim = self.T_dissim_proj(x1_t_embed)
        x1_v_dissim = self.V_dissim_proj(x1_v_embed)
        x1_a_dissim = self.A_dissim_proj(x1_a_embed)

        # concatenate decomposed embeddings for multimodal and unimodal prediction
        x1_sim = torch.cat((x1_t_sim, x1_v_sim, x1_a_sim), dim=-1)
        x1_dissim = torch.cat((x1_t_dissim, x1_v_dissim, x1_a_dissim), dim=-1)
        x1_all = torch.cat((x1_sim, x1_dissim), dim=-1)
        x1_sds = torch.cat((x1_t_sim, x1_v_sim, x1_a_sim, x1_t_dissim, x1_v_dissim, x1_a_dissim), dim=0)

        x_all = x1_all
        label_all = label1
        x_sds = x1_sds
        label1_sds = torch.cat((label1, label1, label1, label1, label1, label1), dim=0)
        label_sds = label1_sds

        if sample2 is not None:
            text2 = tokenize(sample2["text"], device=self.device)
            vision2 = sample2["vision"].clone().detach().to(self.device)
            audio2 = sample2["audio"].clone().detach().to(self.device)
            label2 = [self.class_to_idx[class_name] for class_name in sample2["emotion"]]
            label2 = F.one_hot(torch.tensor(label2), self.num_classes).to(self.device)

            x2_t_embed, x2_v_embed, x2_a_embed = self.encode(text2, vision2, sample2["video_mask"], audio2)

            x2_t_sim = self.T_sim_proj(x2_t_embed)
            x2_v_sim = self.V_sim_proj(x2_v_embed)
            x2_a_sim = self.A_sim_proj(x2_a_embed)
            x2_t_dissim = self.T_dissim_proj(x2_t_embed)
            x2_v_dissim = self.V_dissim_proj(x2_v_embed)
            x2_a_dissim = self.A_dissim_proj(x2_a_embed)

            x2_sim = torch.cat((x2_t_sim, x2_v_sim, x2_a_sim), dim=-1)
            x2_dissim = torch.cat((x2_t_dissim, x2_v_dissim, x2_a_dissim), dim=-1)
            x2_all = torch.cat((x2_sim, x2_dissim), dim=-1)
            x2_sds = torch.cat((x2_t_sim, x2_v_sim, x2_a_sim, x2_t_dissim, x2_v_dissim, x2_a_dissim), dim=0)

            label2_sds = torch.cat((label2, label2, label2, label2, label2, label2), dim=0)

            x_all = torch.cat((x1_all, x2_all), dim=0)
            label_all = torch.cat((label1, label2), dim=0)
            x_sds = torch.cat((x1_sds, x2_sds), dim=0)
            label_sds = torch.cat((label1_sds, label2_sds), dim=0)

        # make multimodal prediction
        pred = self.TVA_decoder(x_all)

        if return_loss:
            # make unimodal prediction
            pred_mono = self.mono_decoder(x_sds)
            # compute multimodal and unimodal loss
            pred_loss = self.criterion(pred, label_all)
            mono_loss = self.criterion(pred_mono, label_sds)
            # compute contrastive loss
            contrastive_loss = 0
            if sample2 is not None:
                # compute NT-Xent contrastive loss over 69 paris (24 positive, 45 negative) for each sample in the batch
                # #inter-positive = 6, #intra-positive = 18
                t1, p = torch.tensor([0, 0, 9, 9, 18, 18,
                                      0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8], device=self.device), \
                    torch.tensor([1, 2, 10, 11, 19, 20,
                                  9, 18, 10, 19, 11, 20, 12, 21, 13, 22, 14, 23, 15, 24, 16, 25, 17, 26],
                                 device=self.device)
                # #inter-negative = 18, #intra-negative = 27
                t2, n = torch.tensor([0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9, 18, 18, 18, 18, 18, 18,
                                      0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8],
                                     device=self.device), \
                    torch.tensor([3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26,
                                  27, 36, 45, 28, 37, 46, 29, 38, 47, 30, 39, 48, 31, 40, 49, 32, 41, 50, 33, 42, 51,
                                  34, 43, 52, 35, 44, 53], device=self.device)
                indices_tuple = (t1, p, t2, n)
                # for each sample, we have 2 positives and 6 negatives
                # two vectors with the same label can form a positive pair
                # this labels matrix is just for easier understanding
                labels = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6,  # x1_t_sim[i], x2_t_sim[8*i: 8*(i+1)]
                                       0, 0, 0, 1, 2, 3, 4, 5, 6,  # x1_v_sim[i], x2_v_sim[8*i: 8*(i+1)]
                                       0, 0, 0, 1, 2, 3, 4, 5, 6,  # x1_a_sim[i], x2_a_sim[8*i: 8*(i+1)]
                                       7, 7, 7, 8, 9, 10, 11, 12, 13,  # x1_t_dissim[i], x2_t_dissim[8*i: 8*(i+1)]
                                       7, 7, 7, 8, 9, 10, 11, 12, 13,  # x1_v_dissim[i], x2_v_dissim[8*i: 8*(i+1)]
                                       7, 7, 7, 8, 9, 10, 11, 12, 13,  # x1_a_dissim[i], x2_a_dissim[8*i: 8*(i+1)]
                                       ])
                for i in range(len(x1_all)):
                    pre_sample_x = []
                    for fea1, fea2 in zip([x1_t_sim, x1_v_sim, x1_a_sim, x1_t_dissim, x1_v_dissim, x1_a_dissim],
                                          [x2_t_sim, x2_v_sim, x2_a_sim, x2_t_dissim, x2_v_dissim, x2_a_dissim]):
                        pre_sample_x.append(torch.cat((fea1[i].unsqueeze(0), fea2[8 * i: 8 * (i + 1)]), dim=0))
                    embeddings = torch.cat(pre_sample_x, dim=0)
                    # don't need to pass in labels if you are already passing in pair/triplet indices
                    contrastive_loss += self.NTXent_loss(embeddings=embeddings, indices_tuple=indices_tuple)
                contrastive_loss /= len(x1_all)
            loss = pred_loss + 0.1 * contrastive_loss + 0.01 * mono_loss
            return loss, pred_loss, contrastive_loss, mono_loss
        else:
            return pred
