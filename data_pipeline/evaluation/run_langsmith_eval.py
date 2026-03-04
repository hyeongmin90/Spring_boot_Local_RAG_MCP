import os
import sys
import uuid
import argparse
from dotenv import load_dotenv

# Run script is inside data_pipeline/evaluation, so project root is two directories up
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from data_pipeline.storage import get_vectorstore, get_hybrid_retriever

# openevals is expected to be available based on previous scripts
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT
from openevals.llm import create_llm_as_judge

def format_docs(docs):
    return "\n\n".join(f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in docs)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline with specific configurations.")
    parser.add_argument(
        "--chunking", 
        nargs="+", 
        choices=["basic", "markdown", "llm"], 
        default=["basic", "markdown", "llm"], 
        help="List of chunking methods to evaluate (e.g. basic markdown llm)"
    )
    parser.add_argument(
        "--retrievers", 
        nargs="+", 
        choices=["dense", "mmr", "hybrid_0.3_0.7", "hybrid_0.5_0.5", "hybrid_0.7_0.3", "hybrid_0.3_0.7_cohere", "hybrid_0.5_0.5_cohere", "hybrid_0.7_0.3_cohere"], 
        default=["hybrid_0.5_0.5", "hybrid_0.5_0.5_cohere"], 
        help="List of retrievers to evaluate"
    )
    return parser.parse_args()

def run_evaluations(args):
    load_dotenv()
    
    # Dataset name in LangSmith (modify this to match your actual dataset)
    dataset_name = os.getenv("LANGSMITH_DATASET_NAME", "sampled_50_questions") 
    print(f"Using LangSmith dataset: {dataset_name}")
    
    # Setup LLM Judges
    judge_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    
    qa_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT, judge=judge_llm, feedback_key="qa", continuous=True, use_reasoning=True
    )
    context_evaluator = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT, judge=judge_llm, feedback_key="context_qa", continuous=True, use_reasoning=True
    )
    retrieval_evaluator = create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT, feedback_key="retrieval_relevance", judge=judge_llm, continuous=True, use_reasoning=True
    )

    def run_qa_eval(run, example):
        # Fallback to empty string if answer is not present in dataset
        reference = example.outputs.get("answer", "") if example.outputs else ""
        return qa_evaluator(
            inputs=example.inputs["question"], 
            reference_outputs=reference, 
            outputs=run.outputs["prediction"]
        )

    def run_context_eval(run, example):
        return context_evaluator(
            context=run.outputs["context"], 
            outputs=run.outputs["prediction"]
        )

    def run_retrieval_eval(run, example):
        return retrieval_evaluator(
            context=run.outputs["context"], 
            inputs=example.inputs["question"]
        )

    evaluators = [run_qa_eval, run_context_eval, run_retrieval_eval]

    # Configurations from args
    chunking_methods = args.chunking
    
    all_retriever_configs = [
        {"name": "dense", "type": "dense"},
        {"name": "mmr", "type": "mmr"},
        {"name": "hybrid_0.3_0.7", "type": "hybrid", "dense_weight": 0.3, "sparse_weight": 0.7, "use_reranker": False},
        {"name": "hybrid_0.5_0.5", "type": "hybrid", "dense_weight": 0.5, "sparse_weight": 0.5, "use_reranker": False},
        {"name": "hybrid_0.7_0.3", "type": "hybrid", "dense_weight": 0.7, "sparse_weight": 0.3, "use_reranker": False},
        {"name": "hybrid_0.3_0.7_cohere", "type": "hybrid", "dense_weight": 0.3, "sparse_weight": 0.7, "use_reranker": True},
        {"name": "hybrid_0.5_0.5_cohere", "type": "hybrid", "dense_weight": 0.5, "sparse_weight": 0.5, "use_reranker": True},
        {"name": "hybrid_0.7_0.3_cohere", "type": "hybrid", "dense_weight": 0.7, "sparse_weight": 0.3, "use_reranker": True},
    ]

    retriever_configs = [c for c in all_retriever_configs if c["name"] in args.retrievers]

    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    prompt = PromptTemplate.from_template(
        """
        You are a Spring Boot Expert RAG Agent.
        Answer user questions accurately using the provided documentation.
        If you cannot find the answer in the search results, admit it honestly.
        Do not include any information not found in the search results.
        Provide clear, code-centric answers where applicable.
        
        Question: {question}
        
        Context: {context}
        """
    )
    
    for chunk_method in chunking_methods:
        collection_name = f"spring_docs_{chunk_method}"
        
        for r_config in retriever_configs:
            experiment_name = f"eval_{chunk_method}_{r_config['name']}"
            print(f"\n{'='*50}")
            print(f"Running evaluation for {experiment_name}")
            print(f"{'='*50}")
            
            def make_predictor(coll_name, config):
                # Retrieve inside the closure to preserve state properly
                def predict_func(inputs: dict):
                    question = inputs["question"]
                    
                    if config["type"] == "dense":
                        vectorstore = get_vectorstore(coll_name)
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
                        docs = retriever.invoke(question)
                    elif config["type"] == "mmr":
                        vectorstore = get_vectorstore(coll_name)
                        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1, "lambda_mult": 0.5})
                        docs = retriever.invoke(question)
                    elif config["type"] == "hybrid":
                        retriever = get_hybrid_retriever(
                            k=1, 
                            collection_name=coll_name, 
                            dense_weight=config["dense_weight"], 
                            sparse_weight=config["sparse_weight"],
                            use_reranker=config["use_reranker"]
                        )
                        # Ensure to use invoke or get_relevant_documents
                        if hasattr(retriever, 'invoke'):
                            docs = retriever.invoke(question)
                        else:
                            docs = retriever.get_relevant_documents(question)
                    
                    context = format_docs(docs)
                    
                    chain = prompt | llm | StrOutputParser()
                    prediction = chain.invoke({"context": context, "question": question})
                    
                    return {
                        "prediction": prediction,
                        "context": context
                    }
                return predict_func

            predictor = make_predictor(collection_name, r_config)

            try:
                evaluate(
                    predictor,
                    data=dataset_name,
                    evaluators=evaluators,
                    experiment_prefix=experiment_name,
                    description=f"Chunking: {chunk_method}, Retriever: {r_config['name']}",
                    max_concurrency=10  # 병렬 처리 추가로 평가 속도 대폭 향상
                )
            except Exception as e:
                print(f"Failed to run evaluation for {experiment_name}: {e}")
                
    print(f"\nCompleted evaluations for combinations: Chunking={chunking_methods}, Retrievers={args.retrievers}")

if __name__ == "__main__":
    args = parse_args()
    run_evaluations(args)
