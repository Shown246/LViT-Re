import torch
import numpy as np
import pandas as pd

#main
def get_embeddings(image_name):
    # df = pd.read_excel("Train_text.xlsx")
    # all_imgs  = df["Image"].astype(str).tolist()
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # for img in all_imgs:
    #     vec = np.load(f"encodings/{img}.npy")
    #     vec = torch.tensor(vec).float().to(device)
    #     print(f"{img}: {vec.shape} on {device}")
    #     print(vec)
    #     print("Done.")
    data = torch.load("encodings/annotation_encodings.pt", map_location='cpu')
    index_map = data["index_map"]
    embeddings = data["embeddings"]    # you can move to GPU per-batch
    lengths = data["lengths"]
    tokens_list = data["tokens"]
    idx = index_map[image_name]
    L = lengths[idx].item()
    emb = embeddings[idx, :L, :].to(device)   # token-level embeddings
    tokens = tokens_list[idx]
    final_result = [(tokens, emb.cpu().numpy())]
    print(final_result)
    return final_result
    # for img_name in all_imgs:
      # idx = index_map[img_name]
      # # do stuff with idx
      # L = lengths[idx].item()
      # emb = embeddings[idx, :L, :].to(device)   # token-level embeddings
      # tokens = tokens_list[idx]
      # final_result = [(tokens, emb.cpu().numpy())]
      # print(final_result)
      
# if __name__ == "__main__":
#     get_embeddings("mask_covid_740.png")