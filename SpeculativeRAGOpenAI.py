from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from time import sleep

@dataclass
class Document:
    """문서를 표현하는 클래스"""
    content: str
    source: str = ""
    score: float = 0.0

# 예시 문서들
SAMPLE_DOCUMENTS = [
    Document(
        content="""Paris is the capital and largest city of France, with an estimated population 
        of 2,102,650 residents as of 2021. The city is the center of the Paris Region or 
        Île-de-France, with an estimated population of 12,262,544 in 2019.""",
        source="Wikipedia - Paris",
    ),
    Document(
        content="""Since the 17th century, Paris has been one of Europe's major centres of 
        finance, diplomacy, commerce, fashion, art, and science. The Paris Region had a GDP of 
        €739 billion in 2019, which is the highest in Europe.""",
        source="Economic Report",
    ),
    Document(
        content="""The Eiffel Tower, built in 1889, is a wrought-iron lattice tower located 
        on the Champ de Mars in Paris, France. It is one of the world's most recognizable 
        landmarks and has become a global symbol of France.""",
        source="Tourist Guide",
    ),
    Document(
        content="""The city of Paris was founded in the 3rd century BC by a Celtic people 
        called the Parisii, who gave the city its name. By the 12th century, Paris was the 
        largest city in the western world.""",
        source="Historical Records",
    ),
    Document(
        content="""Paris is home to many famous museums, including the Louvre Museum, which 
        houses the Mona Lisa and is the world's most visited art museum. The city receives 
        around 30 million tourists annually.""",
        source="Tourism Statistics",
    )
]

class RAGDrafter:
    def __init__(self, model_name: str = "gpt-4"):
        self.model = model_name
        self.client = OpenAI()
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def cluster_documents(self, documents: List[Document], k: int) -> List[List[Document]]:
        # 문서 임베딩
        embeddings = self.embedding_model.encode([doc.content for doc in documents])
        
        # k-means 클러스터링 수행
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # 클러스터별로 문서 그룹화
        clustered_docs = [[] for _ in range(k)]
        for doc, cluster_id in zip(documents, clusters):
            clustered_docs[cluster_id].append(doc)
            
        return clustered_docs

    def generate_draft(self, question: str, documents: List[Document]) -> Tuple[str, str]:
        prompt = f"""Response to the instruction. Also provide rationale for your response.
## Instruction: {question}
## Evidence:
"""
        for i, doc in enumerate(documents, 1):
            prompt += f"[{i}] {doc.content}\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides well-reasoned answers with rationales based on given evidence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            
            try:
                rationale = full_response.split("## Rationale:")[1].split("## Response:")[0].strip()
                answer = full_response.split("## Response:")[1].strip()
            except IndexError:
                rationale = ""
                answer = full_response
                
            return answer, rationale
            
        except Exception as e:
            print(f"Error in generate_draft: {e}")
            sleep(1)
            return "", ""

class RAGVerifier:
    def __init__(self, model_name: str = "gpt-4"):
        self.model = model_name
        self.client = OpenAI()
        
    def compute_score(self, question: str, answer: str, rationale: str) -> float:
        try:
            # self-consistency 점수 계산
            consistency_prompt = f"""Rate how well this answer and rationale pair work together on a scale of 0 to 1.
Question: {question}
Rationale: {rationale}
Answer: {answer}

Provide only a number between 0 and 1."""

            consistency_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates answer quality."},
                    {"role": "user", "content": consistency_prompt}
                ],
                temperature=0
            )
            
            consistency_score = float(consistency_response.choices[0].message.content)
            
            # self-reflection 점수 계산
            reflection_prompt = f"""Question: {question}
Rationale: {rationale}
Answer: {answer}
Do you think the explanation supports the answer? Answer with either 'Yes' or 'No'."""

            reflection_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates answer quality."},
                    {"role": "user", "content": reflection_prompt}
                ],
                temperature=0
            )
            
            reflection_score = 1.0 if "Yes" in reflection_response.choices[0].message.content else 0.0
            
            return consistency_score * reflection_score
            
        except Exception as e:
            print(f"Error in compute_score: {e}")
            sleep(1)
            return 0.0

class SpeculativeRAG:
    def __init__(
        self,
        drafter_model: str = "gpt-4",
        verifier_model: str = "gpt-4",
        num_drafts: int = 3,
        docs_per_draft: int = 2
    ):
        self.drafter = RAGDrafter(drafter_model)
        self.verifier = RAGVerifier(verifier_model)
        self.num_drafts = num_drafts
        self.docs_per_draft = docs_per_draft
        
    def __call__(self, question: str, documents: List[Document]) -> str:
        print(f"Processing question: {question}")
        print(f"Number of input documents: {len(documents)}")
        
        # 문서 클러스터링
        clustered_docs = self.drafter.cluster_documents(documents, k=self.docs_per_draft)
        print(f"Documents clustered into {self.docs_per_draft} groups")
        
        # 초안 생성
        drafts = []
        for i in range(self.num_drafts):
            print(f"\nGenerating draft {i+1}/{self.num_drafts}")
            # 각 클러스터에서 하나의 문서 샘플링
            sampled_docs = [np.random.choice(cluster) for cluster in clustered_docs]
            answer, rationale = self.drafter.generate_draft(question, sampled_docs)
            drafts.append((answer, rationale))
            print(f"Draft {i+1} generated")
        
        # 초안 검증
        scores = []
        for i, (answer, rationale) in enumerate(drafts):
            print(f"\nVerifying draft {i+1}/{self.num_drafts}")
            score = self.verifier.compute_score(question, answer, rationale)
            scores.append(score)
            print(f"Draft {i+1} score: {score:.4f}")
        
        # 최고 점수의 답변 반환
        best_idx = np.argmax(scores)
        print(f"\nSelected draft {best_idx+1} with score {scores[best_idx]:.4f}")
        return drafts[best_idx][0]

def main():
    # OpenAI API 키 설정 (환경 변수로 설정하는 것을 추천)
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # 실제 API 키로 교체 필요
    
    # SpeculativeRAG 초기화
    print("Initializing SpeculativeRAG with GPT-4...")
    rag = SpeculativeRAG()
    
    # 예시 질문
    question = "Tell me about Paris, focusing on its significance as a global city."
    
    print("\nStarting RAG process with GPT-4...")
    answer = rag(question, SAMPLE_DOCUMENTS)
    
    print("\nFinal Results:")
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()