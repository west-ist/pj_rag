import csv
import time
from tqdm import tqdm
from rag_pipeline import init_retriever, rag_answer, non_rag_answer,get_retrieve
from data_loader import load_qa_testset
import config
import re
#评估脚本。包含问题、实际答案、rag答案、非rag答案、hit@k初判（用于判断答案是否命中了给定的知识）、人工判断正确率、retrieval score(余弦相似度）
def run_evaluation():
    print("开始评估\n")
    init_retriever()
    qa_list = load_qa_testset(config.QA_root)
    config.evaluation_root.parent.mkdir(parents=True, exist_ok=True)
    with open(config.evaluation_root, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "question", "answer",
            "non_rag_answer", "rag_answer",
            "hit@k", "correct", "correct_non_rag", "retrieval_scores"
        ])
        for idx, i in enumerate(tqdm(qa_list, desc="评估进度"), 1):
            q = i["question"]
            a = i["answer"]
            #获取回答（加入适当延时避免限流）
            nonrag_ans = non_rag_answer(q)
            time.sleep(0.5)
            rag_ans = rag_answer(q)
            time.sleep(0.5)
            # 检索信息
            retrieved = get_retrieve(q)
            hit = _compute_hit(retrieved, a)
            scores = [round(float(i["similarity"]), 4) for i in retrieved]
            writer.writerow([
                idx, q, a,
                nonrag_ans, rag_ans,
                hit, "", "",   #correct等待人工标注,二编：加入非rag正确率栏
                str(scores)
            ])

    print("评估完成\n")
    print("请手动判断正确率，正确为1，错误为0。并运行failure_analyzer.py生成失败案例报告\n")

#辅助函数，计算hit@k（该概念由llm解释）二编：原先的版本根据空格分割，对英文有效而中文基本无效。重新进行修正后采用正则表达式提取中文字符。
def _compute_hit(retrieved, answer: str)->str:
    if not answer or not retrieved:
        return "0"
    keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}',answer))
    if not keywords:
        return "1" if any(answer in doc['text'] for doc in retrieved) else "0"
    for doc in retrieved:
        text = doc.get('text','')
        if any (j in text for j in keywords):
            return "1"
        return "0"