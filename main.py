#主流程
import random
import numpy as np
from rag_pipeline import init_retriever
from evaluation import run_evaluation
from failure_analyzer import failure_report
import config
import dashscope

dashscope.api_key = config.Dashscope_key
def main():
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    print("欢迎进入个人信息保护法RAG问答系统\n")
    #初始化rag
    print("正在初始化RAG系统\n")
    init_retriever()
    #评估
    print("正在执行评估\n")
    run_evaluation()
    #生成失败案例报告（需先人工标注correct列）
    print("正在生成失败案例报告\n")
    failure_report()

    print("已全部完成\n")
    print("注意：请打开评估文件人工标注correct列（正确为1，错误为0），然后重新运行failure_analyzer.py生成完整报告。\n")

if __name__ == "__main__":
    main()