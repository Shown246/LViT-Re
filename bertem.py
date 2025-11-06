import torch
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
from collections import OrderedDict  

class BertEmbeddingWrapper:
    def __init__(self, model_name='microsoft/BiomedVLP-CXR-BERT-specialized', use_cuda=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

        #device logic
        # if use_cuda and torch.cuda.is_available():
        #     self.device = 'cuda'
        # else:
        #     self.device = 'cpu'
            
        self.device = 'cpu'
        print(f"BertEmbeddingWrapper using device: {self.device}")
        self.model.to(self.device)

    def __call__(self, sentences):
        results = []
        with torch.no_grad():
            # batch tokenize
            inputs = self.tokenizer(
                sentences,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=10
            ).to(self.device)

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state          # [B, T, D]

            # loop per item to match your old return structure
            for i in range(last_hidden.shape[0]):
                token_ids = inputs['input_ids'][i].cpu()
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                token_embeddings = last_hidden[i].cpu().numpy()
                results.append((tokens, token_embeddings))
        return results

# class BertEmbeddingWrapper:
#     """Wrapper for BERT text embeddings using HuggingFace transformers"""
    
#     def __init__(self, model_name='bert-base-uncased', use_cuda=False):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
#         self.model.eval()  # Set to evaluation mode
        
#         # Always use CPU to avoid CUDA fork issues in DataLoader workers
#         self.device = 'cpu'
        
#     def __call__(self, sentences):
#         """
#         Process sentences and return embeddings similar to bert_embedding format
        
#         Args:
#             sentences: List of sentences
            
#         Returns:
#             List of tuples containing (tokens, embeddings)
#         """
#         results = []
        
#         with torch.no_grad():
#             for sentence in sentences:
#                 # Tokenize and encode the sentence
#                 inputs = self.tokenizer(sentence, return_tensors='pt', 
#                                        padding=True, truncation=True, max_length=512)
                
#                 # Get BERT embeddings
#                 outputs = self.model(**inputs)
                
#                 # Extract embeddings from last hidden state
#                 # Shape: [batch_size, sequence_length, hidden_size]
#                 embeddings = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension
                
#                 # Convert to numpy array
#                 embeddings_np = embeddings.numpy()
                
#                 # Get token strings for reference
#                 tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
#                 results.append((tokens, embeddings_np))
        
#         return results

if __name__ == "__main__":
    bert_embedding = BertEmbeddingWrapper()
    df = pd.read_excel("Train_Original.xlsx")
    all_texts = df["Description"].astype(str).tolist()
    all_imgs  = df["Image"].astype(str).tolist()

    results = bert_embedding(all_texts)   # [(tokens, emb_np), ...]
    # print(results)
    #shape
    # for tokens, emb_np in results:
    #     print(f"Tokens: {tokens}")
    #     print(f"Embeddings shape: {emb_np.shape}")
    # determine max length
    lengths = [emb.shape[0] for _, emb in results]
    max_T = max(lengths)
    D = results[0][1].shape[1]
    N = len(results)

    tensor = torch.zeros((N, max_T, D), dtype=torch.float32)
    lengths_arr = torch.zeros(N, dtype=torch.int32)
    tokens_list = [None] * N
    index_map = {}

    for i, (img, (tokens, emb_np)) in enumerate(zip(all_imgs, results)):
        L = emb_np.shape[0]
        tensor[i, :L, :] = torch.from_numpy(emb_np)
        lengths_arr[i] = L
        tokens_list[i] = tokens
        index_map[img] = i

    torch.save({
        "embeddings": tensor,        # [N, max_T, D]
        "lengths": lengths_arr,      # [N]
        "tokens": tokens_list,       # list of token lists
        "index_map": index_map       # img_name -> row
    }, "encodings/annotation_encodings.pt")
    print("Saved embeddings to encodings/annotation_encodings.pt")