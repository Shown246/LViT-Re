import torch
import numpy as np
import pandas as pd

# #main
# def get_embeddings(image_name):
#     # df = pd.read_excel("Train_text.xlsx")
#     # all_imgs  = df["Image"].astype(str).tolist()
    
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
#     data = torch.load("encodings/annotation_encodings.pt", map_location='cpu')
#     index_map = data["index_map"]
#     embeddings = data["embeddings"]    # you can move to GPU per-batch
#     lengths = data["lengths"]
#     tokens_list = data["tokens"]
#     idx = index_map[image_name]
#     L = lengths[idx].item()
#     emb = embeddings[idx, :L, :]# token-level embeddings
#     tokens = tokens_list[idx]
#     final_result = [(tokens, emb.cpu().numpy())]
#     return final_result

# global
DATA = None

def init_embeddings(path="encodings/annotation_encodings.pt"):
    global DATA
    DATA = torch.load(path, map_location="cpu")  # load once

def get_embeddings(image_name):
    # no torch.load here
    data = DATA
    index_map = data["index_map"]
    embeddings = data["embeddings"]
    lengths = data["lengths"]
    tokens_list = data["tokens"]

    idx = index_map[image_name]
    L = lengths[idx].item()
    emb = embeddings[idx, :L, :]
    tokens = tokens_list[idx]
    final_result = [(tokens, emb.cpu().numpy())]
    return final_result
