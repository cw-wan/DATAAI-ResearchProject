# Adapted from https://github.com/XpastaX/ConFEDE/blob/Graph_Main/MOSI/model/net/constrastive/TVA_fusion.py
from modules.confede.decoder.classifier import BaseClassifier
from modules.imagebind.models.imagebind_model import ImageBindModel
from torch import nn
import configs.config_meld as default_config
from modules.imagebind.data import load_and_transform_text as tokenize
from modules.imagebind.models.imagebind_model import ModalityType


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

        uni_fea_dim = int(encoder_fea_dim / 2)

        self.T_sim_proj = Projector(encoder_fea_dim, uni_fea_dim)
        self.V_sim_proj = Projector(encoder_fea_dim, uni_fea_dim)
        self.A_sim_proj = Projector(encoder_fea_dim, uni_fea_dim)

        self.T_dissim_proj = Projector(encoder_fea_dim, uni_fea_dim)
        self.V_dissim_proj = Projector(encoder_fea_dim, uni_fea_dim)
        self.A_dissim_proj = Projector(encoder_fea_dim, uni_fea_dim)

        hidden_size = [uni_fea_dim * 2, uni_fea_dim, int(uni_fea_dim / 2), int(uni_fea_dim / 4),
                       ]

        self.TVA_decoder = BaseClassifier(input_size=uni_fea_dim * 6,
                                          hidden_size=hidden_size,
                                          output_size=1, drop_out=drop_out,
                                          name='TVARegClassifier', )

        self.mono_decoder = BaseClassifier(input_size=uni_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=1, drop_out=drop_out,
                                           name='TVAMonoRegClassifier', )

        self.criterion = nn.MSELoss()

        self.device = config.DEVICE

    def forward(self,
                sample1,
                sample2,
                return_loss=True):
        text1 = sample1["text"]
        vision1 = sample1["vision"].clone().detach().to(self.device)
        audio1 = sample1["audio"].clone().detach().to(self.device)
        token1 = tokenize(text1, device=self.device)

        inputs1 = {
            ModalityType.TEXT: token1,
            ModalityType.VISION: vision1,
            ModalityType.AUDIO: audio1
        }
        embeddings1 = self.imagebind(inputs1)

        x1_t_embed = embeddings1[ModalityType.TEXT]
        x1_v_embed = embeddings1[ModalityType.VISION]
        x1_a_embed = embeddings1[ModalityType.AUDIO]
