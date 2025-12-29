import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# # Add the AdaBins folder to the Python path
# # sys.path.append(os.path.abspath("AdaBins"))
# # from infer import InferenceHelper

# # infer_helper = InferenceHelper(dataset='nyu')

def read_image(img_path, max_retries=5):
    """Read an image while handling IOError with a retry limit."""
    got_img = False
    retries = 0
    while not got_img and retries < max_retries:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print(f"IOError incurred when reading '{img_path}'. Retrying... (Attempt {retries+1})")
            retries += 1
    if not got_img:
        raise IOError(f"Failed to read image {img_path} after {max_retries} attempts.")
    return img

def relabel(labels):
    """Reassigns person IDs to a contiguous range starting from 0 using a dictionary mapping."""
    # Flatten list of lists
    labels_all = [ll for sublist in labels for ll in sublist]
    unique_labels = sorted(set(labels_all) - {-1})
    mapping = {old: new for new, old in enumerate(unique_labels)}
    new_labels = []
    for label in labels:
        temp = []
        for ll in label:
            if ll == -1:
                temp.append(-1)
            else:
                temp.append(mapping[ll])
        new_labels.append(temp)
    num_classes = len(unique_labels)
    return new_labels, num_classes

def relabel_gid(labels):
    """Reassigns group IDs to a contiguous range starting from 0 using a mapping."""
    unique_labels = sorted(set(labels))
    mapping = {old: new for new, old in enumerate(unique_labels)}
    new_labels = [mapping[label] for label in labels]
    return new_labels, len(unique_labels)

class GroupReIDDataset(Dataset):
    def __init__(self, root_dir, transform_g=None, transform_p=None, max_num=5, sigma=10):
        """
        Args:
            root_dir (str): Path to dataset directory.
            transform_g: Transformations for group images.
            transform_p: Transformations for individual person images.
            sigma (int): Sampling rate for 3D space discretization.
            max_num (int): Maximum number of members per group.
        """
        self.root_dir = root_dir
        self.transform = transform_g
        self.transform_p = transform_p
        self.sigma = sigma
        self.data = []
        self.group_count = 0
        self.image_count = 0
        self.max_num = max_num
        
        # Initialize sets to track unique person and group IDs.
        self.unique_person_ids = set()
        self.unique_group_ids = set()
        
        # Load dataset structure.
        for group_id in os.listdir(root_dir):
            group_path = os.path.join(root_dir, group_id)
            if not os.path.isdir(group_path):
                continue
            
            self.group_count += 1
            self.unique_group_ids.add(group_id)
            
            for image_name in os.listdir(group_path):
                if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                
                self.image_count += 1
                
                image_path = os.path.join(group_path, image_name)
                label_path = os.path.join(group_path, image_name.rsplit('.', 1)[0] + '.txt')
                depth_path = os.path.join(group_path, 'depth', image_name.rsplit('.', 1)[0] + '_depth.npy')
                
                if not os.path.exists(label_path):
                    continue
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    person_ids = []
                    bboxes = []
                    for line in lines:
                        parts = line.strip().split()
                        # Skip entries with problematic IDs (e.g., -1 or 7 in some datasets).
                        if parts[0] == '7':
                            continue
                        else:
                            person_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            depth = float(parts[5]) if len(parts) > 5 else None
                            person_ids.append(person_id)
                            bboxes.append((x_center, y_center, width, height, depth))
                            self.unique_person_ids.add(person_id)
                    self.data.append((image_path, depth_path, int(group_id), person_ids, bboxes))

        # Save original IDs.
        self.original_person_ids = [entry[3] for entry in self.data]
        self.original_group_ids = [entry[2] for entry in self.data]
        
        # Apply relabeling.
        self.relabel_flag = True
        if self.relabel_flag:
            self.data_person_ids, self.num_person_classes = relabel(self.original_person_ids)
            self.data_group_ids, self.num_group_classes = relabel_gid(self.original_group_ids)
        else:
            self.data_person_ids = self.original_person_ids
            self.data_group_ids = self.original_group_ids

        self.print_dataset_summary()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, depth_path, _, _, bboxes = self.data[idx]
        person_ids = self.data_person_ids[idx]
        group_id = self.data_group_ids[idx]

        img = read_image(image_path)
        img_width, img_height = img.size

        # Convert YOLO format to pixel coordinates.
        pixel_bboxes = []
        for bbox in bboxes:
            x_center, y_center, width, height, depth = bbox
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            box_width = int(width * img_width)
            box_height = int(height * img_height)
            pixel_bboxes.append((x_min, y_min, box_width, box_height, depth))

        # Load depth map; if fails, raise error.
        try:
            depth_map = np.load(depth_path)
        except Exception as e:
            raise RuntimeError(f"Error loading depth map {depth_path}: {e}")

        # Extract bounding boxes of individuals.
        person_imgs = []
        valid_person_ids = []
        layout_features = []
        for i, bbox in enumerate(pixel_bboxes):
            x, y, w, h, _ = bbox
            cropped_img = img.crop((x, y, x + w, y + h))
            if self.transform_p:
                cropped_img = self.transform_p(cropped_img)
            person_imgs.append(torch.unsqueeze(cropped_img, 0))
            valid_person_ids.append(person_ids[i])
            # Calculate average depth and normalized coordinates.
            avg_depth = self.calculate_average_depth(depth_map, (x, y, w, h))
            # Normalize depth to [0, 1] if necessary.
            avg_depth = (avg_depth - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            normalized_coords = self.calculate_normalized_coordinates((x, y, w, h), img.size)
            layout_features.append((*normalized_coords, avg_depth))

        num_members = len(valid_person_ids)

        # DEBUG: Print shape and one sample's normalized layout features.
        # if idx < 1:  # only for the first sample
        #   print(f"Image size: {img.size}")
        #   print(f"Sample layout features: {layout_features[0]}")
          
        # Padding person images and valid IDs.
        if num_members < self.max_num:
            padded_person_imgs = torch.zeros(self.max_num, 3, 224, 224)
            padded_person_imgs[:num_members] = torch.cat(person_imgs, dim=0)[:num_members]
            padded_valid_person_ids = [-1] * self.max_num
            padded_valid_person_ids[:num_members] = valid_person_ids[:num_members]
            mask = torch.zeros(self.max_num, dtype=torch.bool)
            mask[:num_members] = True
        else:
            padded_person_imgs = torch.cat(person_imgs, dim=0)[:self.max_num]
            padded_valid_person_ids = valid_person_ids[:self.max_num]
            mask = torch.ones(self.max_num, dtype=torch.bool)

        if self.transform:
            img = self.transform(img)

        # Sampling and quantization for layout features.
        sampled_layout_features = self.sample_and_quantize(layout_features)
        # Pad or truncate sampled_layout_features.
        if len(sampled_layout_features) < self.max_num:
            sampled_layout_features += [(-1, -1, -1)] * (self.max_num - len(sampled_layout_features))
        else:
            sampled_layout_features = sampled_layout_features[:self.max_num]
        sampled_layout_features = torch.tensor(sampled_layout_features, dtype=torch.long)

        return img, group_id, padded_person_imgs, padded_valid_person_ids, sampled_layout_features, mask
    
    def calculate_average_depth(self, depth_map, bbox):
        """Calculate the average depth within a bounding box."""
        x, y, w, h = bbox
        depth_region = depth_map[y:y+h, x:x+w]
        return np.mean(depth_region)

    def calculate_normalized_coordinates(self, bbox, img_size):
        """Calculate normalized center coordinates."""
        x, y, w, h = bbox
        img_w, img_h = img_size
        normalized_x = (x + w / 2) / img_w
        normalized_y = (y + h / 2) / img_h
        return normalized_x, normalized_y

    def sample_and_quantize(self, layout_features):
        """Perform sampling and quantization of the 3D space."""
        sampled_features = []
        for feature in layout_features:
            # Multiply by sigma and take integer bins.
            x_bin = min(int(feature[0] * self.sigma), self.sigma - 1)
            y_bin = min(int(feature[1] * self.sigma), self.sigma - 1)
            d_bin = min(max(int(feature[2] * self.sigma), 0), self.sigma - 1)
            sampled_features.append((x_bin, y_bin, d_bin))
        return sampled_features
    
    def get_num_classes(self):
        """Returns the number of person and group classes."""
        return self.num_person_classes, self.num_group_classes
    
    def print_dataset_summary(self):
        """Prints dataset statistics."""
        print("\nDataset Summary")
        print("----------------------------")
        print(f"Total Groups: {self.group_count}")
        print(f"Total Images: {self.image_count}")
        print(f"Total Entries: {len(self.data)}")
        print(f"Unique Person IDs: {self.num_person_classes}")
        print(f"Unique Group IDs: {self.num_group_classes}")
        print("----------------------------\n")

    # def estimate_depth(self, img):
    #     """Estimate depth map using a pre-trained model."""
    #     bin_centers, predicted_depth = infer_helper.predict_pil(img)
    #     original_size = img.size
    #     # Convert predicted_depth to a tensor if it's not already
    #     predicted_depth_tensor = torch.tensor(predicted_depth[0][0])
    #     # Resize the predicted depth to the original image size
    #     prediction = torch.nn.functional.interpolate(
    #         predicted_depth_tensor.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
    #         size=(original_size[1], original_size[0]),  # (height, width)
    #         mode="bicubic",
    #         align_corners=False,
    #     ).squeeze()  # Remove batch and channel dimensions
    #     depth_map = prediction.cpu().numpy()
    #     depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize
    #     return depth_map