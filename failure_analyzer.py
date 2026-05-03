#收集失败案例
import pandas as pd
import config
#找出evaluation中correct为0的行，将其写入
def failure_report():
    eval = config.evaluation_root
    if not eval.exists():
        print("评估文件不存在\n")
        return
    read = pd.read_csv(eval)
    if 'correct' not in read.columns:
        print("未判断正确情况\n")
        return
    failure = read[read['correct'] == 0]
    total = len(read)
    failed = len(failure)
    config.failure_root.parent.mkdir(parents=True, exist_ok=True)
    with open(config.failure_root,'w',encoding='utf-8') as f:
        f.write(f"失败案例汇总\n")
        f.write(f"共{total}个问题，其中{failed}个失败\n")
        f.write(f"失败案例如下\n")
        for i,row in failure.iterrows():
            f.write(f"案例{i+1}\n")
            f.write(f"问题:{row['question']}\n")
            f.write(f"答案:{row['answer']}\n")
            f.write(f"RAG回答:{row['rag_answer']}\n")
            f.write(f"非RAG回答:{row['non_rag_answer']}\n")
            f.write(f"失败原因（手动填写）：")
            f.write("\n")
    print("已生成失败案例报告，需手动填写失败原因\n")

if __name__ == '__main__':
    failure_report()