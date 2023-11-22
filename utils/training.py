from modules.confede.TVA_fusion import TVAFusion
from dataloaders.MELD_dataloader import dataloaderMELD
from configs import config_meld, naive_roberta_config, naive_imagebind_adapter_config, \
    imagebind_adapter_contrastive_config, roberta_contrastive_config
from modules import NaiveRoBERTa, NaiveImagebindAdapter, ImagebindAdapterContrastive, ContrastiveRoBERTa
import torch
import transformers
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils.common import write_log

EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


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


def eval_tva_fusion(model, config):
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
        mf1 = f1_score(truth, pred, labels=np.arange(7), average='macro')
        model.train()
    return acc, wf1, mf1


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

    # weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(params=optimizer_grouped_params, lr=lr, amsgrad=False, )
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              num_warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )
    model.to(device)

    all_loss = torch.tensor(0)
    pred_loss = torch.tensor(0)
    contrastive_loss = torch.tensor(0)
    for epoch in range(1, total_epoch + 1):
        _update_matrix(train_dataloader, model)
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample1, in enumerate(bar):
            bar.set_description(
                "Epoch:%d|All_loss:%s|Loss:%s|Contrastive_loss:%s" % (
                    epoch, all_loss.item(), pred_loss.item(), contrastive_loss.item()))

            optimizer.zero_grad()

            idx_list = sample1['index']
            sample2 = train_dataloader.dataset.sample(idx_list)

            pred_result, all_loss, pred_loss, contrastive_loss, mono_loss = model(sample1, sample2)

            all_loss.backward()
            optimizer.step()
            scheduler.step()

        print("EVAL valid")
        acc, wf1, mf1 = eval_tva_fusion(model, config=config_meld)
        log = "Epoch {}, Accuracy {}, Weighted F1 Score {}, Macro F1 Score {}".format(epoch, acc, wf1, mf1)
        print(log)
        write_log(log, path='TVA_Fusion_train.log')
        model.save_model(epoch=epoch)


def test_tva_fusion(load_epoch):
    # build the model
    model = TVAFusion()
    model.load_model(load_checkpoint_epoch=load_epoch)
    device = config_meld.DEVICE
    model.to(device)

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
        mf1 = f1_score(truth, pred, labels=np.arange(7), average='macro')

    log = "Test Epoch {}, Accuracy {}, Weighted F1 Score {}, Macro F1 Score {}".format(load_epoch, acc, wf1, mf1)
    print(log)
    print(confusion_matrix)


def eval_naive_roberta(model):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    with torch.no_grad():
        model.eval()
        eval_dataloader = dataloaderMELD(datapath=naive_roberta_config.Path.data, subset="dev", batch_size=32,
                                         shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(eval_dataloader)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truths.append(np.array(label))
            pred = model(sample, return_loss=False)
            pred = torch.argmax(pred, dim=-1)
            predictions.append(pred.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions, labels=np.arange(7), average='weighted')
        mf1 = f1_score(truths, predictions, labels=np.arange(7), average='macro')
    return acc, f1, mf1


def train_naive_roberta():
    device = naive_roberta_config.device
    # load training parameters
    batch_size = naive_roberta_config.DownStream.batch_size
    learning_rate = naive_roberta_config.DownStream.learning_rate
    warm_up = naive_roberta_config.DownStream.warm_up
    total_epoch = naive_roberta_config.DownStream.total_epoch
    decay = naive_roberta_config.DownStream.decay

    # init model
    model = NaiveRoBERTa()
    model.to(device)

    # init dataloader
    train_dataloader = dataloaderMELD(datapath=naive_roberta_config.Path.data, subset="train",
                                      batch_size=batch_size,
                                      shuffle=False)

    # weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # init optimizer
    optimizer = torch.optim.AdamW(params=optimizer_grouped_params, lr=learning_rate, amsgrad=False)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )

    # train
    loss = 0
    acc, f1, mf1 = eval_naive_roberta(model)
    print("Before training, Accuracy {}, Weighted F1 Score {}, Macro F1 Score {}".format(acc, f1, mf1))
    for epoch in range(1, total_epoch + 1):
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample, in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:%s" % (epoch, loss))
            loss, pred = model(sample)
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # evaluate
        acc, f1, mf1 = eval_naive_roberta(model)
        log = "Epoch {}, Accuracy {}, Weighted F1 Score {}, Macro F1 Score {}".format(epoch, acc, f1, mf1)
        print(log)
        write_log(log, path='naive_roberta_train.log')
        # save model
        model.save_model(epoch)


def test_naive_roberta(load_epoch):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    device = naive_roberta_config.device
    # load trained model
    model = NaiveRoBERTa()
    model.load_model(load_epoch)
    model.to(device)

    # confusion matrix
    confusion_matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)))

    with torch.no_grad():
        model.eval()
        test_dataloader = dataloaderMELD(datapath=naive_roberta_config.Path.data, subset="test",
                                         batch_size=32,
                                         shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(test_dataloader)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truths.append(np.array(label))
            pred = model(sample, return_loss=False)
            pred = torch.argmax(pred, dim=-1)
            predictions.append(pred.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        # update the confusion matrix
        for idx, pred_class in enumerate(predictions):
            confusion_matrix[truths[idx]][pred_class] += 1

        # compute the weighted F1
        acc = accuracy_score(truths, predictions)
        wf1 = f1_score(truths, predictions, labels=np.arange(7), average='weighted')
        mf1 = f1_score(truths, predictions, labels=np.arange(7), average='macro')

    log = "Test Epoch {}, Accuracy {}, Weighted F1 Score {}, Macro F1 Score {}".format(load_epoch, acc, wf1, mf1)
    print(log)
    print(confusion_matrix.astype('int32'))


def eval_naive_imagebind_adapter(model):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    with torch.no_grad():
        model.eval()
        eval_dataloader = dataloaderMELD(datapath=naive_imagebind_adapter_config.Path.data, subset="dev", batch_size=32,
                                         shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(eval_dataloader)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truths.append(np.array(label))
            pred = model(sample, return_loss=False)
            pred = torch.argmax(pred, dim=-1)
            predictions.append(pred.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions, labels=np.arange(7), average='weighted')
    return acc, f1


def train_naive_imagebind_adapter():
    device = naive_roberta_config.device
    # load training parameters
    batch_size = naive_imagebind_adapter_config.DownStream.batch_size
    learning_rate = naive_imagebind_adapter_config.DownStream.learning_rate
    warm_up = naive_imagebind_adapter_config.DownStream.warm_up
    total_epoch = naive_imagebind_adapter_config.DownStream.total_epoch
    decay = naive_imagebind_adapter_config.DownStream.decay

    # init model
    model = NaiveImagebindAdapter()
    model.load_model(load_pretrain=True)
    model.freeze_imagebind()
    model.to(device)

    # init dataloader
    train_dataloader = dataloaderMELD(datapath=naive_imagebind_adapter_config.Path.data,
                                      subset="train",
                                      batch_size=batch_size,
                                      shuffle=False)

    # weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # init optimizer
    optimizer = torch.optim.AdamW(params=optimizer_grouped_params, lr=learning_rate, amsgrad=False)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )

    # train
    loss = 0
    for epoch in range(1, total_epoch + 1):
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample, in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:%s" % (epoch, loss))
            pred, loss = model(sample)
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # evaluate
        acc, f1 = eval_naive_imagebind_adapter(model)
        log = "Epoch {}, Accuracy {}, F1 Score {}".format(epoch, acc, f1)
        print(log)
        write_log(log, path='naive_imagebind_adapter_train.log')
        # save model
        model.save_model(epoch)


def test_naive_imagebind_adapter(load_epoch):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    device = naive_imagebind_adapter_config.device
    # load trained model
    model = NaiveImagebindAdapter()
    model.load_model(load_pretrain=False, load_checkpoint_epoch=load_epoch)
    model.to(device)

    # confusion matrix
    confusion_matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)))

    with torch.no_grad():
        model.eval()
        test_dataloader = dataloaderMELD(datapath=naive_imagebind_adapter_config.Path.data,
                                         subset="test",
                                         batch_size=32,
                                         shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(test_dataloader)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truths.append(np.array(label))
            pred = model(sample, return_loss=False)
            pred = torch.argmax(pred, dim=-1)
            predictions.append(pred.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        # update the confusion matrix
        for idx, pred_class in enumerate(predictions):
            confusion_matrix[truths[idx]][pred_class] += 1

        # compute the weighted F1
        acc = accuracy_score(truths, predictions)
        wf1 = f1_score(truths, predictions, labels=np.arange(7), average='weighted')

    log = "Test Epoch {}, Accuracy {}, F1 Score {}".format(load_epoch, acc, wf1)
    print(log)
    print(confusion_matrix.astype('int32'))


def eval_imagebind_adapter_contrastive(model):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    with torch.no_grad():
        model.eval()
        eval_dataloader = dataloaderMELD(datapath=imagebind_adapter_contrastive_config.Path.data,
                                         subset="dev",
                                         batch_size=32,
                                         shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(eval_dataloader)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truths.append(np.array(label))
            pred = model(sample, return_loss=False)
            pred = torch.argmax(pred, dim=-1)
            predictions.append(pred.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions, labels=np.arange(7), average='weighted')
    return acc, f1


def train_imagebind_adapter_contrastive():
    device = imagebind_adapter_contrastive_config.device
    # load training parameters
    batch_size = imagebind_adapter_contrastive_config.DownStream.batch_size
    learning_rate = imagebind_adapter_contrastive_config.DownStream.learning_rate
    warm_up = imagebind_adapter_contrastive_config.DownStream.warm_up
    total_epoch = imagebind_adapter_contrastive_config.DownStream.total_epoch
    decay = imagebind_adapter_contrastive_config.DownStream.decay

    # init model
    model = ImagebindAdapterContrastive()
    model.load_model(load_pretrain=True)
    model.freeze_imagebind()
    model.to(device)

    # init dataloader
    train_dataloader = dataloaderMELD(datapath=imagebind_adapter_contrastive_config.Path.data,
                                      subset="train",
                                      batch_size=batch_size,
                                      shuffle=False)

    # weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # init optimizer
    optimizer = torch.optim.AdamW(params=optimizer_grouped_params, lr=learning_rate, amsgrad=False)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )

    # train
    loss = torch.tensor(0)
    pred_loss = torch.tensor(0)
    contrastive_loss = torch.tensor(0)
    for epoch in range(1, total_epoch + 1):
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample, in enumerate(bar):
            bar.set_description(
                "Epoch:%d|All Loss:%s, Pred Loss: %s, Contrastive Loss:%s" % (
                    epoch, loss.item(), pred_loss.item(), contrastive_loss.item()))
            sample2 = train_dataloader.dataset.contrastive_sample(sample["index"])
            pred, loss, pred_loss, contrastive_loss = model(sample=sample, sample2=sample2, return_loss=True)
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # evaluate
        acc, f1 = eval_imagebind_adapter_contrastive(model)
        log = "Epoch {}, Accuracy {}, F1 Score {}".format(epoch, acc, f1)
        print(log)
        write_log(log, path='imagebind_adapter_contrastive_train.log')
        # save model
        model.save_model(epoch)


def test_imagebind_adapter_contrastive(load_epoch):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    device = imagebind_adapter_contrastive_config.device
    # load trained model
    model = ImagebindAdapterContrastive()
    model.load_model(load_pretrain=False, load_checkpoint_epoch=load_epoch)
    model.to(device)

    # confusion matrix
    confusion_matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)))

    with torch.no_grad():
        model.eval()
        test_dataloader = dataloaderMELD(datapath=imagebind_adapter_contrastive_config.Path.data,
                                         subset="test",
                                         batch_size=32,
                                         shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(test_dataloader)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truths.append(np.array(label))
            pred = model(sample, return_loss=False)
            pred = torch.argmax(pred, dim=-1)
            predictions.append(pred.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        # update the confusion matrix
        for idx, pred_class in enumerate(predictions):
            confusion_matrix[truths[idx]][pred_class] += 1

        # compute the weighted F1
        acc = accuracy_score(truths, predictions)
        wf1 = f1_score(truths, predictions, labels=np.arange(7), average='weighted')

    log = "Test Epoch {}, Accuracy {}, F1 Score {}".format(load_epoch, acc, wf1)
    print(log)
    print(confusion_matrix.astype('int32'))


def eval_contrastive_roberta(model):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    with torch.no_grad():
        model.eval()
        eval_dataloader = dataloaderMELD(datapath=roberta_contrastive_config.Path.data, subset="dev", batch_size=32,
                                         shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(eval_dataloader)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truths.append(np.array(label))
            pred = model(sample, return_loss=False)
            pred = torch.argmax(pred, dim=-1)
            predictions.append(pred.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions, labels=np.arange(7), average='weighted')
        mf1 = f1_score(truths, predictions, labels=np.arange(7), average='macro')
    return acc, f1, mf1


def train_contrastive_roberta():
    device = roberta_contrastive_config.device
    # load training parameters
    batch_size = roberta_contrastive_config.DownStream.batch_size
    learning_rate = roberta_contrastive_config.DownStream.learning_rate
    warm_up = roberta_contrastive_config.DownStream.warm_up
    total_epoch = roberta_contrastive_config.DownStream.total_epoch
    decay = roberta_contrastive_config.DownStream.decay

    # init model
    model = ContrastiveRoBERTa()
    model.to(device)

    # init dataloader
    train_dataloader = dataloaderMELD(datapath=naive_roberta_config.Path.data, subset="train",
                                      batch_size=batch_size,
                                      shuffle=False)

    # weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # init optimizer
    optimizer = torch.optim.AdamW(params=optimizer_grouped_params, lr=learning_rate, amsgrad=False)
    scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                          num_warmup_steps=int(
                                                                              warm_up * (len(train_dataloader))),
                                                                          num_training_steps=total_epoch * len(
                                                                              train_dataloader), )

    # train
    loss = torch.tensor(0)
    pred_loss = torch.tensor(0)
    contrastive_loss = torch.tensor(0)
    acc, f1, mf1 = eval_contrastive_roberta(model)
    print("Before training, Accuracy {}, Weighted F1 Score {}, Macro F1 {}".format(acc, f1, mf1))
    for epoch in range(1, total_epoch + 1):
        model.train()
        bar = tqdm(train_dataloader)
        for index, sample, in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:%s, Pred Loss:%s, Contrastive Loss:%s" % (
                epoch, loss.item(), pred_loss.item(), contrastive_loss.item()))
            sample2 = train_dataloader.dataset.contrastive_sample(sample["index"])
            pred, loss, pred_loss, contrastive_loss = model(sample, sample2)
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        # evaluate
        acc, f1, mf1 = eval_contrastive_roberta(model)
        log = "Epoch {}, Accuracy {}, Weighted F1 Score {}, Macro F1 Score {}".format(epoch, acc, f1, mf1)
        print(log)
        write_log(log, path='contrastive_roberta_train.log')
        # save model
        model.save_model(epoch)


def test_contrastive_roberta(load_epoch):
    class_to_idx = {class_name: idx for idx, class_name in enumerate(EMOTION_LABELS)}
    device = roberta_contrastive_config.device
    # load trained model
    model = ContrastiveRoBERTa()
    model.load_model(load_epoch)
    model.to(device)

    # confusion matrix
    confusion_matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)))

    with torch.no_grad():
        model.eval()
        test_dataloader = dataloaderMELD(datapath=roberta_contrastive_config.Path.data, subset="test",
                                         batch_size=32,
                                         shuffle=False)
        predictions = []
        truths = []
        bar = tqdm(test_dataloader)
        for index, sample in enumerate(bar):
            label = [class_to_idx[class_name] for class_name in sample["emotion"]]
            truths.append(np.array(label))
            pred = model(sample, return_loss=False)
            pred = torch.argmax(pred, dim=-1)
            predictions.append(pred.cpu().detach().numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        # update the confusion matrix
        for idx, pred_class in enumerate(predictions):
            confusion_matrix[truths[idx]][pred_class] += 1

        # compute the weighted F1
        acc = accuracy_score(truths, predictions)
        wf1 = f1_score(truths, predictions, labels=np.arange(7), average='weighted')
        mf1 = f1_score(truths, predictions, labels=np.arange(7), average='macro')

    log = "Test Epoch {}, Accuracy {}, F1 Score {}, Macro F1 Score {}".format(load_epoch, acc, wf1, mf1)
    print(log)
    print(confusion_matrix.astype('int32'))


MODELS = {
    "naive-roberta": {
        "train": train_naive_roberta,
        "test": test_naive_roberta
    },
    "naive-imagebind-adapter": {
        "train": train_naive_imagebind_adapter,
        "test": test_naive_imagebind_adapter
    },
    "imagebind-adapter-contrastive": {
        "train": train_imagebind_adapter_contrastive,
        "test": test_imagebind_adapter_contrastive
    },
    "contrastive-roberta": {
        "train": train_contrastive_roberta,
        "test": test_contrastive_roberta
    },
    "confede-imagebind-adapter": {
        "train": train_tva_fusion,
        "test": test_tva_fusion
    }
}
