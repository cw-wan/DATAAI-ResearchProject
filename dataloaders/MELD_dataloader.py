import os
from torch.utils.data import Dataset
from utils import VideoLoader
import pandas as pd
import pickle
import logging

logging.basicConfig(filename="dataloader.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.WARNING)


class DataloaderMELD(Dataset):
    def __init__(self,
                 path,
                 subset,
                 fps):
        self.path = {
            'train': os.path.join(path, 'train'),
            'dev': os.path.join(path, 'dev'),
            'test': os.path.join(path, 'test')
        }
        assert subset in ['train', 'dev', 'test']
        self.data_path = os.path.join(self.path[subset], subset + ".pkl")
        if not os.path.exists(self.data_path):
            videoloader = VideoLoader(224, 16, fps)
            video_path = os.path.join(self.path[subset], subset + "_splits")
            csv_path = os.path.join(self.path[subset], subset + "_sent_emo.csv")
            csv_data = pd.read_csv(csv_path)
            csv_data = csv_data.loc[:, ["Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"]]
            self.data = []
            idx = 0
            for _, row in csv_data.iterrows():
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
                        self.data.append(utt)
                    except Exception as e:
                        logging.warning("{} {} {}".format(subset, video, e))
                else:
                    logging.warning("{} {} NOT FOUND".format(subset, video))
                idx += 1
                print("\rPreprocessing MELD {} {}%".format(subset, round(100.0 * idx / csv_data.shape[0], 2)), end="")
            print("\nSaving and loading MELD ...")
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
        else:
            print("Loading MELD " + subset + " ...")
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        return self.data[idx]
