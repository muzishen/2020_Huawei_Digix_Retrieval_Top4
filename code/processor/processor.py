import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda.amp import autocast as autocast, GradScaler
from utils.reranking import re_ranking
import datetime
import torch.nn.functional as F

def do_train(cfg,
             model,
             train_loader,
             optimizer,
             scheduler,
             loss_fn):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    if device:
        model.to(device)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # train
    scaler = GradScaler()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):

            optimizer.zero_grad()
            if cfg.INPUT.AUGMIX:
                bs = img[0].size(0)
                images_cat = torch.cat(img, dim = 0).to(device) # [3 * batch, 3, 32, 32]
                target = vid.to(device)
                with autocast():
                    logits, feat = model(images_cat, target)
                    logits_orig, logits_augmix1, logits_augmix2 = logits[:bs], logits[bs:2*bs], logits[2*bs:]
                    loss = loss_fn (logits_orig, feat, target)
                    p_orig, p_augmix1, p_augmix2 = F.softmax(logits_orig, dim = -1), F.softmax(logits_augmix1, dim = -1), F.softmax(logits_augmix2, dim = -1)

                    # Clamp mixture distribution to avoid exploding KL divergence
                    p_mixture = torch.clamp((p_orig + p_augmix1 + p_augmix2) / 3., 1e-7, 1).log()
                    loss += 12 * (F.kl_div(p_mixture, p_orig, reduction='batchmean') +
                                    F.kl_div(p_mixture, p_augmix1, reduction='batchmean') +
                                    F.kl_div(p_mixture, p_augmix2, reduction='batchmean')) / 3.
            else:
                img = img.to(device)
                target = vid.to(device)
                with autocast():
                    if cfg.MODEL.CHANNEL_HEAD:
                        score, feat, channel_head_feature = model(img, target)
                        #print(feat.shape, channel_head_feature.shape)
                        loss = loss_fn(score, feat, channel_head_feature, target)

                    else:
                        score, feat = model(img, target)
                        loss = loss_fn(score, feat, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        scheduler.step()
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1 - cosine

def do_inference(
        cfg,
        model,
        val_loader,
        num_query,
        query_name,
        gallery_name
):

    model.eval()
    model.cuda()
    features = torch.FloatTensor().cuda()
    for (input_img, pid, cid) in val_loader:
        input_img = input_img.cuda()
        input_img_mirror = input_img.flip(dims=[3])
        outputs = model(input_img)
        outputs_mirror = model(input_img_mirror)
        f = outputs + outputs_mirror
        # flip
        features = torch.cat((features, f), 0)

    if cfg.TEST.RE_RANKING:
        feats = torch.nn.functional.normalize(features, dim=1, p=2)
        qf = feats[:num_query]
        gf = feats[num_query:]
        ranking_parameter = cfg.TEST.RE_RANKING_PARAMETER
        k1 = ranking_parameter[0]
        k2 = ranking_parameter[1]
        lambda_value = ranking_parameter[2]
        distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lambda_value)
    else:
        qf = features[:num_query]
        gf = features[num_query:]
        distmat = cosine_dist(qf, gf)
        distmat = distmat.cpu().numpy()
        
    #np.save(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT) , distmat)
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    max_10_indices = indices[:, :10]
    res_dict = dict()
    for q_idx in range(num_q):
        filename = query_name[q_idx].split("/")[-1]
        max_10_files = [gallery_name[i].split("/")[-1] for i in max_10_indices[q_idx]]
        res_dict[filename] = max_10_files
    with open('%s/submission.csv' % cfg.OUTPUT_DIR, 'w') as file:
        for k, v in res_dict.items():
            writer_string = "%s,{%s,%s,%s,%s,%s,%s,%s,%s,%s,%s}\n"%(k, v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9])
            file.write(writer_string)
    file.close()