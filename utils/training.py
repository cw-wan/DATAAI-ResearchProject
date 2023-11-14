from modules.confede.TVA_fusion import TVAFusion
from modules.baseline.baseline_model import Baseline
from dataloaders.MELD_dataloader import dataloaderMELD
from configs import config_meld, config_baseline
import torch
import transformers
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.common import write_log


EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def build_tva_fusion(epoch):
    model = TVAFusion()
    model.load_model(load_checkpoint_epoch=epoch)
    device = config_meld.DEVICE
    model.to(device)
    return model


def build_baseline(epoch):
    model = Baseline()
    model.load_model(load_checkpoint_epoch=epoch)
    model.to(config_baseline.DEVICE)
    return model



MODEL_BUILDER = {
    "TVA_Fusion": build_tva_fusion,
    "Baseline": build_baseline
}


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


def eval_model(model, config):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    with torch.no_grad():
        model.eval()
        eval_data = dataloaderMELD(datapath=config.MELD.Path.raw_data_path,
                                   subset="dev",
                                   batch_size=32,
                                   shuffle=False)
        pred = []
        truth = []
        bar = tqdm(eval_data)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truth.append(np.array(label))
            pred_result = model(sample, return_loss=False)[0]
            pred_result = pred_result.to(torch.device('cpu'))
            pred.append(pred_result.numpy())
        pred = np.concatenate(pred)
        truth = np.concatenate(truth)
        acc = accuracy_score(truth, pred)
        wf1 = f1_score(truth, pred, labels=np.arange(7), average='weighted')
        model.train()
    return acc, wf1


def train_tva_fusion():
    model = TVAFusion()
    model.load_model(load_pretrain=True)

    model.freeze_imagebind()

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

    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=False, )
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              num_warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )
    model.to(device)

    all_loss = 0
    pred_loss = 0
    contrastive_loss = 0
    for epoch in range(1, total_epoch + 1):
        if epoch % 2 == 1:  # originally: epoch % 2 == 1, but since imagebing frozen, no need to update
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
        acc, wf1 = eval_model(model, config=config_meld)
        log = "Epoch {}, Accuracy {}, F1 Score {}".format(epoch, acc, wf1)
        print(log)
        write_log(log, path='TVA_Fusion_train.log')
        if epoch > 1:
            model.save_model(epoch=epoch)


def train_baseline():
    model = Baseline()
    model.load_model(load_pretrain=True)
    model.freeze_imagebind()

    device = config_baseline.DEVICE
    batch_size = config_baseline.MELD.Downstream.batch_size
    lr = config_baseline.MELD.Downstream.lr
    total_epoch = config_baseline.MELD.Downstream.epoch
    num_warm_up = config_baseline.MELD.Downstream.num_warm_up

    train_dataloader = dataloaderMELD(datapath=config_baseline.MELD.Path.raw_data_path,
                                      subset="train",
                                      batch_size=batch_size,
                                      shuffle=True)

    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=False, )
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              num_warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )
    model.to(device)

    loss = 0
    for epoch in range(1, total_epoch + 1):
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample, in enumerate(bar):
            bar.set_description(
                "Epoch:%d|Loss:%s" % (epoch, loss))

            optimizer.zero_grad()
            pred_result, loss = model(sample)
            loss.backward()
            optimizer.step()
            scheduler.step()

        print("EVAL valid")
        acc, wf1 = eval_model(model, config=config_baseline)
        log = "Epoch {}, Accuracy {}, F1 Score {}".format(epoch, acc, wf1)
        print(log)
        write_log(log, path='baseline_train.log')
        model.save_model(epoch=epoch)


def test(model, epoch):
    # build the model
    model = MODEL_BUILDER[model](epoch)

    # confusion matrix
    confusion_matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)))

    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    with torch.no_grad():
        model.eval()
        eval_data = dataloaderMELD(datapath=config_meld.MELD.Path.raw_data_path,
                                   subset="test",
                                   batch_size=32,
                                   shuffle=False)
        pred = []
        truth = []
        bar = tqdm(eval_data)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truth.append(np.array(label))
            pred_result = model(sample, return_loss=False)[0]
            pred_result = pred_result.to(torch.device('cpu'))
            pred.append(pred_result.numpy())
        pred = np.concatenate(pred)
        truth = np.concatenate(truth)

        # update the confusion matrix
        for idx, pred_class in enumerate(pred):
            confusion_matrix[truth[idx]][pred_class] += 1

        # compute the weighted F1
        acc = accuracy_score(truth, pred)
        wf1 = f1_score(truth, pred, labels=np.arange(7), average='weighted')

    log = "Test Epoch {}, Accuracy {}, F1 Score {}".format(epoch, acc, wf1)
    print(log)
    print(confusion_matrix)
