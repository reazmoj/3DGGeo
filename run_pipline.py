import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from three_d_transformer import ThreeDTransformer
from group_reid_dataset import GroupReIDDataset
from evaluation import evaluate_model, infer, visualize_retrieval, extract_features, plot_losses
from training_pipeline import train
from sampler import GroupPKBatchSampler, RandomIdentitySampler

def main():
    parser = argparse.ArgumentParser(description="Train, Evaluate, and Infer Group Re-Identification Model")
    # Dataset directories.
    parser.add_argument("--data-root", type=str, default="dataset", help="Root directory of the dataset")
    parser.add_argument("--train-dir", type=str, default="train_test/train", help="Sub-folder for training data")
    parser.add_argument("--query-dir", type=str, default="train_test/test/query", help="Sub-folder for query data")
    parser.add_argument("--gallery-dir", type=str, default="train_test/test/gallery", help="Sub-folder for gallery data")
    # General parameters.
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "infer"], default="train", help="Mode: train, eval, or infer")
    # Training parameters.
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--warmup", type=int, default=5, help="warm up each epochs")
    
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    # Model parameters.
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of transformer heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--sigma", type=int, default=10, help="Discretization parameter for layout tokens")
    parser.add_argument("--model-path", type=str, default="model.pth", help="Path to save/load the model")
    parser.add_argument('--num-instances', type=int, default=4, help="number of instances per identity")
    # Inference parameters.
    parser.add_argument("--query-image", type=str, default="", help="Path to query image for inference")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define transforms.
    transform_g = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3),
        transforms.Normalize(
            mean=[0.485,0.456,0.406], 
            std=[0.229,0.224,0.225])
    ])

    transform_p = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),             
        transforms.RandomErasing(p=0.3),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if args.mode == "train":
        train_dataset = GroupReIDDataset(
            root_dir=f"{args.data_root}/{args.train_dir}",
            transform_g=transform_g,
            transform_p=transform_p,
            sigma=args.sigma
        )
        person_num_classes, group_num_classes = train_dataset.get_num_classes()
        model = ThreeDTransformer(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            person_num_classes=person_num_classes,
            group_num_classes=group_num_classes,
            sigma=args.sigma
        ).to(device)
        
        batch_sampler = GroupPKBatchSampler(
            train_dataset,
            P=8,
            K=4,
            drop_last=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True
        )

        query_loader = None
        gallery_loader = None

        if args.query_dir and args.gallery_dir:
            query_dataset = GroupReIDDataset(
                root_dir=f"{args.data_root}/{args.query_dir}",
                transform_g=transform_g,
                transform_p=transform_p,
                sigma=args.sigma
            )
            query_loader = DataLoader(
                query_dataset, batch_size=args.test_batch_size,
                shuffle=False, num_workers=args.workers
            )

            gallery_dataset = GroupReIDDataset(
                root_dir=f"{args.data_root}/{args.gallery_dir}",
                transform_g=transform_g,
                transform_p=transform_p,
                sigma=args.sigma
            )
            gallery_loader = DataLoader(
                gallery_dataset, batch_size=args.test_batch_size,
                shuffle=False, num_workers=args.workers
            )

        loss_history = train(model, train_loader, query_loader, gallery_loader, args.epochs, args.lr, args.weight_decay, device, save_path=args.model_path)
        plot_losses(loss_history)
        print("Training complete.")
  
    elif args.mode == "eval":
        query_dataset = GroupReIDDataset(
            root_dir=f"{args.data_root}/{args.query_dir}",
            transform_g=transform_g,
            transform_p=transform_p,
            sigma=args.sigma
        )
        gallery_dataset = GroupReIDDataset(
            root_dir=f"{args.data_root}/{args.gallery_dir}",
            transform_g=transform_g,
            transform_p=transform_p,
            sigma=args.sigma
        )
        person_num_classes, group_num_classes = query_dataset.get_num_classes()
        model = ThreeDTransformer(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            person_num_classes=person_num_classes,
            group_num_classes=group_num_classes,
            sigma=args.sigma
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        query_loader = DataLoader(
            query_dataset, batch_size=args.test_batch_size,
            shuffle=False, num_workers=args.workers
        )
        gallery_loader = DataLoader(
            gallery_dataset, batch_size=args.test_batch_size,
            shuffle=False, num_workers=args.workers
        )
        r1, r5, r10, mAP = evaluate_model(model, query_loader, gallery_loader, device)
        print("Evaluation Results:")
        print(f"Rank-1: {r1:.4f}, Rank-5: {r5:.4f}, Rank-10: {r10:.4f}, mAP: {mAP:.4f}")
    
    elif args.mode == "infer":
        if args.query_image == "":
            print("Please provide a query image path with --query-image")
            return
        gallery_dataset = GroupReIDDataset(
            root_dir=f"{args.data_root}/{args.gallery_dir}",
            transform_g=transform_g,
            transform_p=transform_p,
            sigma=args.sigma
        )
        person_num_classes, group_num_classes = gallery_dataset.get_num_classes()
        model = ThreeDTransformer(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            person_num_classes=person_num_classes,
            group_num_classes=group_num_classes,
            sigma=args.sigma
        ).to(device)
        gallery_loader = DataLoader(
            gallery_dataset, batch_size=args.test_batch_size,
            shuffle=False, num_workers=args.workers
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        # Use feature extraction mode for inference.
        query_feat = infer(model, args.query_image, transform_p, args.sigma, device)
        gallery_feats, gallery_labels = extract_features(model, gallery_loader, device)
        gallery_feats_np = gallery_feats.numpy()
        dists = np.linalg.norm(gallery_feats_np - query_feat, axis=1)
        sorted_indices = np.argsort(dists)
        retrieved_imgs = [gallery_dataset.data[idx][0] for idx in sorted_indices[:5]]
        correct_flags = [False] * len(retrieved_imgs)  # Placeholder flags.
        query_img = Image.open(args.query_image).convert("RGB")
        visualize_retrieval(query_img, retrieved_imgs, correct_flags)
    
if __name__ == "__main__":
    main()

