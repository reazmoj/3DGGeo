import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from three_d_transformer import ThreeDTransformer
from group_reid_dataset import GroupReIDDataset

def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for batch in dataloader:
            images, group_ids, person_imgs, _, sampled_layout_features, _ = batch
            person_imgs = person_imgs.to(device)
            sampled_layout_features = sampled_layout_features.to(device)
            # Call forward with return_feature=True to extract embeddings.
            group_feats = model(person_imgs, sampled_layout_features, return_feature=True)
            features_list.append(group_feats.cpu())
            labels_list.append(group_ids.cpu())
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels

def compute_metrics(query_feats, query_labels, gallery_feats, gallery_labels, use_cosine=False):
    if use_cosine:
        query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
        gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1, keepdims=True)
        dists = 1 - np.dot(query_feats, gallery_feats.T)
    else:
        dists = np.linalg.norm(query_feats[:, np.newaxis] - gallery_feats[np.newaxis, :], axis=2)
    
    num_query = query_feats.shape[0]
    rank1, rank5, rank10 = 0, 0, 0
    aps = []
    for i in range(num_query):
        q_label = query_labels[i]
        sorted_idx = np.argsort(dists[i])
        sorted_labels = gallery_labels[sorted_idx]
        correct = (sorted_labels == q_label).astype(np.int32)
        if correct[:1].sum() > 0:
            rank1 += 1
        if correct[:5].sum() > 0:
            rank5 += 1
        if correct[:10].sum() > 0:
            rank10 += 1
        
        num_relevant = correct.sum()
        if num_relevant == 0:
            aps.append(0)
        else:
            tmp_cumsum = np.cumsum(correct)
            precisions = [tmp_cumsum[k] / (k + 1) for k in range(len(correct)) if correct[k]]
            aps.append(np.mean(precisions))
    
    rank1_rate = rank1 / num_query
    rank5_rate = rank5 / num_query
    rank10_rate = rank10 / num_query
    mAP = np.mean(aps)
    return rank1_rate, rank5_rate, rank10_rate, mAP

def evaluate_model(model, query_loader, gallery_loader, device):
    query_feats, query_labels = extract_features(model, query_loader, device)
    gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
    return compute_metrics(query_feats.numpy(), query_labels.numpy(), gallery_feats.numpy(), gallery_labels.numpy())

def infer(model, query_image, transform_p, sigma, device):
    if not isinstance(query_image, Image.Image):
        try:
            query_image = Image.open(query_image).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error opening query image: {e}")
    person_img = transform_p(query_image)
    person_imgs = person_img.unsqueeze(0).unsqueeze(0)
    norm_coords = (0.5, 0.5, 0.5)
    sampled = [min(int(coord * sigma), sigma - 1) for coord in norm_coords]
    sampled_layout_features = torch.tensor(sampled, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    person_imgs = person_imgs.to(device)
    sampled_layout_features = sampled_layout_features.to(device)
    with torch.no_grad():
        group_feat = model(person_imgs, sampled_layout_features, return_feature=True)
    return group_feat.cpu().numpy().flatten()

def visualize_retrieval(query_image, retrieved_images, correct_flags):
    if not isinstance(query_image, Image.Image):
        query_image = Image.open(query_image).convert("RGB")
    num_retrieved = len(retrieved_images)
    fig, axes = plt.subplots(1, num_retrieved + 1, figsize=(4*(num_retrieved+1), 4))
    axes[0].imshow(query_image)
    axes[0].set_title("Query")
    axes[0].axis("off")
    for i, img in enumerate(retrieved_images):
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        axes[i+1].imshow(img)
        title = "Correct" if correct_flags[i] else "Incorrect"
        axes[i+1].set_title(f"Rank {i+1}\n{title}")
        axes[i+1].axis("off")
    plt.tight_layout()
    plt.show()
    
def plot_losses(loss_history):
    epochs = range(1, len(loss_history["loss"]) + 1)

    plt.figure(figsize=(12, 8))

    plt.plot(epochs, loss_history["loss"], label="Total Loss", linewidth=2)
    plt.plot(epochs, loss_history["cp"], label="Person CE")
    plt.plot(epochs, loss_history["tp"], label="Person Triplet")
    plt.plot(epochs, loss_history["cg"], label="Group CE")
    plt.plot(epochs, loss_history["tg"], label="Group Triplet")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("losses.png")
    plt.close() 