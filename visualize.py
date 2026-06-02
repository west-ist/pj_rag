#二编：加入rag与非rag回答准确率的可视化对比
import pandas as pd
import matplotlib.pyplot as plt
import config


def accuracy_comparison(csv_path=config.evaluation_root, save_path="outputs/accuracy_comparison.png"):
    df = pd.read_csv(csv_path, encoding='utf-8')
    rag_acc = df['correct'].mean()*100
    non_rag_acc = df['correct_non_rag'].mean()*100
    plt.bar(['RAG','non-RAG'],[rag_acc,non_rag_acc])
    plt.ylim(0,100)
    plt.ylabel('accuracy(%)')
    plt.title('RAG vs non-RAG accuracy')
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    accuracy_comparison()