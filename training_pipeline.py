import torch
import torch.optim as optim
from group_reid_losses import GroupReIDLoss
import logging
from torch.cuda.amp import autocast, GradScaler
from evaluation import evaluate_model

# Set up basic logging.
logging.basicConfig(level=logging.INFO)

def adjust_learning_rate(optimizer, epoch, base_lr, warmup_epochs):
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / warmup_epochs
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def freeze_backbone(model, freeze=True):
    for name, param in model.named_parameters():
        if "vit" in name:
            param.requires_grad = not freeze
              
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()

    total_loss = 0.0
    total_cp = 0.0
    total_tp = 0.0
    total_cg = 0.0
    total_tg = 0.0

    for batch in dataloader:
        group_imgs, group_ids, person_imgs, valid_person_ids, sampled_layout_features, mask = batch

        person_imgs = person_imgs.to(device)
        sampled_layout_features = sampled_layout_features.to(device)
        group_ids = group_ids.to(device)

        if not torch.is_tensor(valid_person_ids):
            valid_person_ids = torch.stack(
                [torch.tensor(x, dtype=torch.long) for x in valid_person_ids]
            ).to(device)
        else:
            valid_person_ids = valid_person_ids.to(device)

        mask = mask.to(device)

        optimizer.zero_grad()

        person_feats, person_logits, group_feats, group_logits = model(
            person_imgs, sampled_layout_features
        )

        B, N, D = person_feats.shape
        person_feats = person_feats.view(B * N, D)
        person_logits = person_logits.view(B * N, -1)
        person_labels = valid_person_ids.view(-1)
        person_mask = (mask.view(-1)) & (person_labels >= 0)

        loss, L_cp, L_tp, L_cg, L_tg = criterion(
            person_feats, person_logits,
            group_feats, group_logits,
            person_labels, group_ids,
            person_mask=person_mask,
            return_components=True
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_cp += L_cp.item()
        total_tp += L_tp.item()
        total_cg += L_cg.item()
        total_tg += L_tg.item()

    scheduler.step()

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "cp": total_cp / num_batches,
        "tp": total_tp / num_batches,
        "cg": total_cg / num_batches,
        "tg": total_tg / num_batches,
    }


loss_history = {
    "loss": [],
    "cp": [],
    "tp": [],
    "cg": [],
    "tg": [],
}
           
def train(model, train_loader, query_loader,gallery_loader, num_epochs, lr, weight_decay, device, save_path="best_model.pth", warmup_epochs=5, freeze_epochs=15):
    criterion = GroupReIDLoss(alpha=0.5, beta=1.0, margin=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)

    best_map = 0.0

    for epoch in range(num_epochs):
        
        #backbone freezing
        if epoch < freeze_epochs:
            freeze_backbone(model, freeze=True)
        else:
            freeze_backbone(model, freeze=False)

        # warmup LR
        adjust_learning_rate(optimizer, epoch, lr, warmup_epochs)
        
        stats = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        for k in loss_history:
            loss_history[k].append(stats[k])

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"L={stats['loss']:.4f} | "
            f"CP={stats['cp']:.4f} | TP={stats['tp']:.4f} | "
            f"CG={stats['cg']:.4f} | TG={stats['tg']:.4f}"
        )
        

        if query_loader is not None and epoch % 5 == 0:
            r1, r5, r10, mAP = evaluate_model(model, query_loader, gallery_loader, device)
            # logging.info(f"Validation: R1={r1:.4f}, R5={r5:.4f}, R10={r10:.4f}, mAP={mAP:.4f}")
            print(f"Validation: R1={r1:.4f}, R5={r5:.4f}, R10={r10:.4f}, mAP={mAP:.4f}")
            
            # Save best model
            if mAP > best_map:
                best_map = mAP
                torch.save(model.state_dict(), save_path)
                print(f"New best model saved with mAP: {mAP:.4f}")
    
    return loss_history

