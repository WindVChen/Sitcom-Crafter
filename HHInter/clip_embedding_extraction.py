# Processing a dataset of annotations: extract the CLIP text embeddings of the txt annotations.

import os
import torch
import numpy as np
from tqdm import tqdm
import clip
from HHInter.global_path import get_dataset_path


def extract_clip_emb(file_path):
    with open(file_path, "r") as f:
        annots = f.readlines()
        # In each file, there are multiple text annotations.
        for i in range(len(annots)):
            annots[i] = annots[i].replace("\n", "")
            # Truncate the text to 77 characters.
            if len(annots[i]) > 77:
                annots[i] = annots[i][:77]

    text = clip.tokenize(annots).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features


def compare_scores(text, emb_folder=os.path.join(get_dataset_path(), "Inter-X/clip_embs")):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Calculate the input text's clip embeddings and compare with all the pre-calculated clip_embeds and select the top1's name.
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        max_score = -1
        max_name = ""
        max_content = ""
        for file in tqdm(os.listdir(emb_folder), total=len(os.listdir(emb_folder))):
            if file.endswith(".npy"):
                emb = np.load(os.path.join(emb_folder, file))
                scores = (text_features.cpu().numpy() @ emb.T).flatten()
                for id, score in enumerate(scores):
                    if score > max_score:
                        max_score = score
                        max_name = file.split(".")[0]
                        max_content = open(os.path.join(emb_folder.replace('clip_embs', 'annots'), file.split(".")[0] + ".txt"), "r").readlines()[id]

    # get the hand pose of the corresponding motion file.
    data1 = np.load(os.path.join(emb_folder.replace('clip_embs', 'motions'), max_name, "P1.npz"))
    data2 = np.load(os.path.join(emb_folder.replace('clip_embs', 'motions'), max_name, "P2.npz"))
    pose_hand_1 = np.concatenate([data1['pose_lhand'], data1['pose_rhand']], axis=1)
    pose_hand_2 = np.concatenate([data2['pose_lhand'], data2['pose_rhand']], axis=1)

    return max_name, max_score, max_content, pose_hand_1, pose_hand_2


if __name__ == '__main__':
    is_test = False

    if not is_test:
        data_path = os.path.join(get_dataset_path(), "Inter-X/annots")
        emb_folder = os.path.join(get_dataset_path(), "Inter-X/clip_embs")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        if not os.path.exists(emb_folder):
            os.makedirs(emb_folder)

        for file in tqdm(os.listdir(data_path), total=len(os.listdir(data_path))):
            if file.endswith(".txt"):
                file_path = os.path.join(data_path, file)
                emb = extract_clip_emb(file_path)
                np.save(os.path.join(emb_folder, file.split(".")[0] + ".npy"), emb.cpu().numpy())
    else:
        # Test Retrieval.
        text = "both they approach each other."
        max_name, max_score, max_content, _, _ = compare_scores(text)
        print(max_name, max_score, max_content)