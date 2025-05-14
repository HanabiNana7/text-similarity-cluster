from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


class TextClusterer:
    def __init__(
        self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ) -> None:
        """
        初始化 TextClusterer，載入指定的 SentenceTransformer 模型。

        Args:
            model_name (str, optional): SentenceTransformer 模型名稱。
                預設為 "paraphrase-multilingual-MiniLM-L12-v2"。
                模型說明可參見：
                https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        """
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def texts_embedding(self, texts: List[str]) -> List[np.ndarray]:
        """
        將一組文字轉換為對應的向量。

        Args:
            texts (List[str]): 輸入的文字列表。

        Returns:
            List[np.ndarray]: 對應的嵌入向量列表。
        """
        return self.model.encode(texts, convert_to_tensor=False)

    def cluster_texts(
        self, texts: List[str], dbscan_eps: float, min_dbscan_samples: int
    ) -> np.ndarray:
        """
        對輸入文字進行向量化後，使用 DBSCAN 演算法進行分群。

        Args:
            texts (List[str]): 要進行分群的文字列表。
            dbscan_eps (float): DBSCAN 中的最大距離參數 epsilon。
            min_dbscan_samples (int): DBSCAN 中每個群所需的最小樣本數。

        Returns:
            np.ndarray: 每筆文字所屬的群集標籤，-1 表示未分群（即離群點）。
        """
        embeddings: List[np.ndarray] = self.texts_embedding(texts)

        dbscan = DBSCAN(
            eps=dbscan_eps,
            min_samples=min_dbscan_samples,
            metric="cosine",
        )
        labels: np.ndarray = dbscan.fit_predict(embeddings)

        return labels
