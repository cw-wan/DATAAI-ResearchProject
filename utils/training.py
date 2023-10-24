from modules.confede.TVA_fusion import TVAFusion
from dataloaders.MELD_dataloader import dataloaderMELD
from configs import config_meld
import torch
import transformers
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.common import write_log


def _update_matrix(dataloader, model):
    with torch.no_grad():
        model.eval()
        train_data = dataloaderMELD(datapath=config_meld.MELD.Path.raw_data_path,
                                    subset="train",
                                    batch_size=32,
                                    shuffle=False)
        T, V, A = [], [], []
        print('Collecting New embeddings ...')
        bar = tqdm(train_data)
        for index, sample in enumerate(bar):
            _, _T, _V, _A = model(sample, return_loss=False)
            T.append(_T)
            V.append(_V)
            A.append(_A)
        T = torch.cat(T, dim=0).to(torch.device('cpu'))
        V = torch.cat(V, dim=0).to(torch.device('cpu'))
        A = torch.cat(A, dim=0).to(torch.device('cpu'))
        X = torch.cat((T, V, A), dim=-1)
        X /= X.norm(dim=-1, keepdim=True)
        sim_matrix = X @ X.T
        print('Updating Similarity Matrix ...')
        dataloader.dataset.update_matrix(sim_matrix.numpy())
        model.train()
    return sim_matrix


def eval_tva_fusion(model):
    emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(emotion_labels)}
    with torch.no_grad():
        model.eval()
        eval_data = dataloaderMELD(datapath=config_meld.MELD.Path.raw_data_path,
                                   subset="dev",
                                   batch_size=32,
                                   shuffle=False)
        pred = []
        truth = []
        bar = tqdm(eval_data)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truth.append(np.array(label))
            pred_result, _, _, _ = model(sample, return_loss=False)
            pred_result = pred_result.to(torch.device('cpu'))
            pred.append(pred_result.numpy())
        pred = np.concatenate(pred)
        truth = np.concatenate(truth)
        print(pred)
        print(truth)
        print(pred.shape)
        print(truth.shape)
        acc = accuracy_score(truth, pred)
        wf1 = f1_score(truth, pred, labels=np.arange(7), average='weighted')
        model.train()
    return acc, wf1


def train_tva_fusion():
    model = TVAFusion()
    model.load_model(load_pretrain=True)

    device = config_meld.DEVICE
    batch_size = config_meld.MELD.Downstream.batch_size
    lr = config_meld.MELD.Downstream.lr
    total_epoch = config_meld.MELD.Downstream.epoch
    decay = config_meld.MELD.Downstream.decay
    num_warm_up = config_meld.MELD.Downstream.num_warm_up

    train_dataloader = dataloaderMELD(datapath=config_meld.MELD.Path.raw_data_path,
                                      subset="train",
                                      batch_size=batch_size,
                                      shuffle=True)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=lr, amsgrad=False, )
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              num_warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )
    model.to(device)

    all_loss = 0
    pred_loss = 0
    contrastive_loss = 0
    save_period = 5  # save checkpoint every 5 epochs
    for epoch in range(1, total_epoch + 1):
        if epoch % 2 == 1:
            _update_matrix(train_dataloader, model)
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample1, in enumerate(bar):
            bar.set_description(
                "Epoch:%d|All_loss:%s|Loss:%s|Contrastive_loss:%s" % (epoch, all_loss, pred_loss, contrastive_loss))

            optimizer.zero_grad()

            idx_list = sample1['index']
            sample2 = train_dataloader.dataset.sample(idx_list)

            pred_result, all_loss, pred_loss, contrastive_loss, mono_loss = model(sample1, sample2)

            all_loss.backward()
            optimizer.step()
            scheduler.step()

        print("EVAL valid")
        acc, wf1 = eval_tva_fusion(model)
        log = "Epoch {}, Accuracy {}, F1 Score {}".format(epoch, acc, wf1)
        print(log)
        write_log(log, path='TVA_Fusion_train.log')
        if epoch % save_period == 0:
            model.save_model(epoch=epoch)
