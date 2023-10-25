import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import VideoLoader
import pandas as pd
import pickle
import logging
import numpy as np

logging.basicConfig(filename="dataloader.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.WARNING)

np.random.seed(64)


class DatasetMELD(Dataset):
    def __init__(self,
                 path,
                 subset,
                 fcnt=4,
                 fps=1):
        self.path = {
            'train': os.path.join(path, 'train'),
            'dev': os.path.join(path, 'dev'),
            'test': os.path.join(path, 'test')
        }
        assert subset in ['train', 'dev', 'test']
        self.data_path = os.path.join(self.path[subset], subset + ".pkl")
        if not os.path.exists(self.data_path):
            videoloader = VideoLoader(224, fcnt, fps)
            video_path = os.path.join(self.path[subset], subset + "_splits")
            csv_path = os.path.join(self.path[subset], subset + "_sent_emo.csv")
            csv_data = pd.read_csv(csv_path)
            csv_data = csv_data.loc[:, ["Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"]]
            self.data = []
            idx = 0
            for _i, row in csv_data.iterrows():
                utt = dict()
                utt["text"] = row["Utterance"]
                utt["speaker"] = row["Speaker"]
                utt["emotion"] = row["Emotion"]
                utt["sentiment"] = row["Sentiment"]
                utt["dialogue_id"] = row["Dialogue_ID"]
                utt["utterance_id"] = row["Utterance_ID"]
                video = os.path.join(video_path,
                                     "dia" + str(row["Dialogue_ID"]) + "_utt" + str(row["Utterance_ID"]) + ".mp4")
                if os.path.exists(video):
                    try:
                        utt["vision"], utt["video_mask"], utt["audio"] = videoloader.load(video)
                        utt["index"] = idx
                        idx += 1
                        self.data.append(utt)
                    except Exception as e:
                        logging.warning("{} {} {}".format(subset, video, e))
                else:
                    logging.warning("{} {} NOT FOUND".format(subset, video))
                print("\rPreprocessing MELD {} {}%".format(subset, round(100.0 * (_i + 1) / csv_data.shape[0], 2)),
                      end="")
            print("\nSaving and loading MELD ...")
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
        else:
            print("Loading MELD " + subset + " ...")
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        # sim_matrix[i, j]: cosine similarity between data[i] and data[j]
        self.sim_matrix = np.zeros((len(self.data), len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def update_matrix(self, matrix):
        self.sim_matrix = matrix

    def sample(self, idx):
        """
        Implementation of ConFEDE sampling algorithm.
        :param idx: list of sample1 data ids
        """
        z = int(len(self.data) / 4)
        samples = {}
        for key in self.data[0].keys():
            samples[key] = []
        for i in idx:
            sorted_idx = np.argsort(self.sim_matrix[i, :])
            similar_sample_set = []
            dissimilar_sample_set = []
            for j in sorted_idx:
                if i == j:
                    continue
                if self.data[i]["emotion"] == self.data[j]["emotion"]:
                    similar_sample_set.append(j)
                else:
                    dissimilar_sample_set.append(j)
            neighbour = np.random.permutation(similar_sample_set[-z:])[:2]
            outlier1 = np.random.permutation(dissimilar_sample_set[:z])[:3]
            outlier2 = np.random.permutation(dissimilar_sample_set[-z:])[:3]
            samples_idx = np.concatenate((neighbour, outlier1, outlier2))
            for _id in samples_idx:
                for key in samples.keys():
                    samples[key].append(self.data[_id][key])
        for key in ["vision", "video_mask", "audio"]:
            samples[key] = torch.stack(samples[key])
        return samples


def dataloaderMELD(datapath, subset, batch_size, shuffle=True):
    dataset = DatasetMELD(datapath, subset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
