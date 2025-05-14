from collections import defaultdict

from src.text2vec_cluster import TextClusterer


def main():
    texts = [
        "進擊的巨人",
        "犬夜叉",
        "半妖的夜叉姬",
        "火影忍者",
        "慕留人 - 火影新世代 -",
    ]

    clusterer = TextClusterer()
    labels = clusterer.cluster_texts(
        texts,
        dbscan_eps=0.3,
        min_dbscan_samples=2,
    )

    clustered_titles = defaultdict(list)
    for text, label in zip(texts, labels):
        clustered_titles[label].append(text)

    # 輸出每一群的結果
    for cluster_id, group in clustered_titles.items():
        if cluster_id != -1:
            print(f"\n 群組 {cluster_id + 1}:")
            for text in group:
                print(f" - {text}")
        else:
            print("\n 噪聲資料: 無法歸類的項目")
            for text in group:
                print(f" - {text}")


if __name__ == "__main__":
    main()
