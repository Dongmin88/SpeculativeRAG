from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import torch
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

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
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
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
        # 프롬프트 구성
        prompt = f"""Response to the instruction. Also provide rationale for your response.
## Instruction: {question}
## Evidence:
"""
        for i, doc in enumerate(documents, 1):
            prompt += f"[{i}] {doc.content}\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 응답에서 근거와 답변 추출
        try:
            rationale = response.split("## Rationale:")[1].split("## Response:")[0].strip()
            answer = response.split("## Response:")[1].strip()
        except IndexError:
            rationale = ""
            answer = response
            
        return answer, rationale

class RAGVerifier:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def compute_score(self, question: str, answer: str, rationale: str) -> float:
        # self-consistency 점수 계산
        prompt = f"Question: {question}\nRationale: {rationale}\nAnswer: {answer}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids)
            logits = outputs.logits
            
        # 확률 점수 계산
        probs = F.softmax(logits, dim=-1)
        answer_tokens = self.tokenizer(answer, add_special_tokens=False).input_ids
        rationale_tokens = self.tokenizer(rationale, add_special_tokens=False).input_ids
        
        # 답변과 근거에 대한 점수 계산
        score_answer = torch.mean(torch.tensor([probs[0, i, token].item() 
                                              for i, token in enumerate(answer_tokens)]))
        score_rationale = torch.mean(torch.tensor([probs[0, i, token].item() 
                                                 for i, token in enumerate(rationale_tokens)]))
        
        # self-reflection 점수 계산
        reflection_prompt = f"{prompt}\nDo you think the explanation supports the answer? (Yes or No)"
        reflection_inputs = self.tokenizer(reflection_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            reflection_outputs = self.model.generate(
                **reflection_inputs,
                max_new_tokens=5,
                temperature=0.0
            )
            
        reflection = self.tokenizer.decode(reflection_outputs[0], skip_special_tokens=True)
        reflection_score = 1.0 if "Yes" in reflection else 0.0
        
        # 최종 점수 결합
        final_score = (score_answer + score_rationale) / 2 * reflection_score
        return final_score.item()

class SpeculativeRAG:
    def __init__(
        self,
        drafter_model: str = "mistralai/Mistral-7B-v0.1",
        verifier_model: str = "mistralai/Mistral-7B-v0.1",
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
    # SpeculativeRAG 초기화
    print("Initializing SpeculativeRAG...")
    rag = SpeculativeRAG()
    
    # 예시 질문과 문서
    question = "Tell me about Paris, focusing on its significance as a global city."
    
    print("\nStarting RAG process...")
    answer = rag(question, SAMPLE_DOCUMENTS)
    
    print("\nFinal Results:")
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()