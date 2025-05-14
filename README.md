# text2vec-cluster

用於將文字轉換為向量（Embeddings），並透過分群演算法找出語意上相似的句子群組。

---

## 功能簡介

- 使用 `sentence-transformers` 將文字轉換為語意向量
- 使用 `DBSCAN` 進行語意分群
- 適用於：關鍵字分群、意圖分類、語意搜尋前處理等

---

## 主要相依套件

- `sentence-transformers`：文字轉向量模型
- `scikit-learn`：分群與相似度演算法

---

## 專案緣起

日前看到手邊凌亂的資料夾 / 檔案，也開始思考過去有沒有相關的經驗可以幫我依照系列分類好這些檔案，也因此有了該項獨立的小專案。
後續可以延伸 os 移動檔案或其他相關的功能，若讀者剛好有需求，歡迎隨時複製及修改。

---

## 參考資源

- 使用到的 [文字轉向量模型](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- 使用到的 [分群演算法](https://myapollo.com.tw/blog/dbscan/)

---
