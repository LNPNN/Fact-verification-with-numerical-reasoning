from tqdm import tqdm
import torch
import nltk
from sentence_transformers import SentenceTransformer
import re
import math

# Load SBERT model hỗ trợ tiếng Việt
model_name = "MiuN2k3/ViWikiSBert-fine-tuning"
model = SentenceTransformer(model_name)

nltk.download('punkt')

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def calculate_bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.75):
    doc_len = len(doc_tokens)
    avg_doc_len = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens)
    scores = []
    for doc in doc_tokens:
        doc_freq = {}
        for token in doc:
            if token in doc_freq:
                doc_freq[token] += 1
            else:
                doc_freq[token] = 1
        score = 0
        for token in query_tokens:
            if token in doc_freq:
                idf = math.log((len(doc_tokens) - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5) + 1.0)
                tf = doc_freq[token] * (k1 + 1) / (doc_freq[token] + k1 * (1 - b + b * doc_len / avg_doc_len))
                score += idf * tf
        scores.append(score)
    return scores

def encode_sbert(text):
    return model.encode(text, convert_to_tensor=True)

def ER_BM25_Sbert(paragraph, claim, top_best_envidence):
    sentences = nltk.sent_tokenize(paragraph)
    doc_tokens = [tokenize(sentence) for sentence in sentences]

    # Tokenize the claim
    claim_tokens = tokenize(claim)

    # Calculate BM25 scores for each sentence in the paragraph
    scores = calculate_bm25_score(claim_tokens, doc_tokens, k1=1.5, b=0.75)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_best_envidence]

    # Get the most relevant sentences based on BM25 scores
    most_similar_sentences = [sentences[i] for i in top_k_indices]

    similarities = []
    encoded_claim = encode_sbert(claim).unsqueeze(0)
    for sent in most_similar_sentences:
        encoded_sentence = encode_sbert(sent).unsqueeze(0)
        similarity = torch.nn.functional.cosine_similarity(encoded_claim, encoded_sentence)
        similarities.append(similarity.item())

    # Get the top sentences based on SBERT similarity scores
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_best_envidence]
    top_evidence = [most_similar_sentences[i] for i in top_indices]

    return top_evidence

# Example usage
# premise = "Ở tuổi 26, Al Capone thành ông chủ quyền lực, lãnh thổ kinh doanh rượu lậu vươn tới Canada với sự bảo vệ của chính trị gia. Al Capone sở hữu các nhà máy bia, nhà thổ, tụ điểm cờ bạc, đồn điền và thành trùm tội phạm ở Chicago."
# hypo = "Al Capone trở thành người giàu có ở tuổi 26 bởi vì ông sở hữu rất nhiều loại hình kinh doanh."
# num = 3
# print(ER_BM25_Sbert(premise, hypo, num))

