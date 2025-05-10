from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


class TextClusterer:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """
        預設使用 paraphrase-multilingual-MiniLM-L12-v2 模型轉換文字向量，可參閱：
        https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

        Args:
            model_name (str, optional): SentenceTransformer 模型名稱
        """
        self.model = SentenceTransformer(model_name)

    def texts_embedding(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

    def cluster_texts(self, texts):
        embeddings = self.texts_embedding(texts)

        # 使用 DBSCAN 進行分群
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
        # 可調整 eps 和 min_samples 參數
        labels = dbscan.fit_predict(embeddings)

        return labels
