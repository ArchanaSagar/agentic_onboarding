"""Agentic onboarding assistant (LangChain starter)

Features:
- Interactive: asks domain and role
- Loads local docs from ./docs (md/txt/py) and chunks them
- Builds a vector retriever with Chroma/OpenAI embeddings if available
- Falls back to a simple keyword search if vector store is not available
- Produces a role-specific summary, finds learning materials, falls back to contacts
- Generates a short multiple-choice quiz and gives feedback

Notes:
- Requires OPENAI_API_KEY in the environment.
- Optional: install chromadb to enable vector search (faster/better retrieval).
"""

import os
import argparse
import json
from typing import List, Dict

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
	# prefer vector-backed retriever if available
	from langchain.embeddings import OpenAIEmbeddings
	from langchain.vectorstores import Chroma
	VECTOR_BACKEND = "chroma"
except Exception:
	OpenAIEmbeddings = None
	Chroma = None
	VECTOR_BACKEND = None


class OnboardingAgent:
	def __init__(self, model_name: str = "gpt-4o-mini", docs_dir: str = "docs"):
		self.model_name = model_name
		self.llm = ChatOpenAI(model_name=model_name, temperature=0.2)
		self.docs_dir = docs_dir
		self.documents: List[Dict] = []
		self.retriever = None

	def load_local_documents(self):
		docs = []
		if not os.path.isdir(self.docs_dir):
			print(f"No local docs directory at {self.docs_dir}; continuing without local docs.")
			self.documents = []
			return

		for root, _, files in os.walk(self.docs_dir):
			for fn in files:
				if fn.lower().endswith((".md", ".txt", ".py")):
					path = os.path.join(root, fn)
					try:
						with open(path, "r", encoding="utf-8") as f:
							text = f.read()
					except Exception:
						continue
					docs.append({"path": path, "text": text})

		splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
		chunks = []
		for d in docs:
			parts = splitter.split_text(d["text"])
			for i, p in enumerate(parts):
				chunks.append({"source": d["path"], "page": i, "text": p})

		self.documents = chunks

	def build_retriever(self):
		if VECTOR_BACKEND == "chroma" and OpenAIEmbeddings is not None and Chroma is not None and self.documents:
			texts = [d["text"] for d in self.documents]
			metadatas = [{"source": d["source"], "page": d["page"]} for d in self.documents]
			try:
				embeddings = OpenAIEmbeddings()
				chroma = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas)
				self.retriever = chroma.as_retriever(search_kwargs={"k": 4})
				print("Using Chroma vector retriever for document search.")
				return
			except Exception:
				self.retriever = None

		self.retriever = None

	def keyword_search(self, query: str, top_k: int = 4) -> List[Dict]:
		q = query.lower().split()
		scored = []
		for d in self.documents:
			text = d["text"].lower()
			score = sum(text.count(token) for token in q)
			if score > 0:
				scored.append((score, d))
		scored.sort(key=lambda x: x[0], reverse=True)
		return [d for _, d in scored[:top_k]]

	def retrieve_materials(self, query: str, top_k: int = 4) -> List[Dict]:
		if self.retriever is not None:
			try:
				docs = self.retriever.get_relevant_documents(query)
				results = []
				for doc in docs[:top_k]:
					results.append({"source": getattr(doc, "metadata", {}).get("source"), "text": doc.page_content})
				return results
			except Exception:
				pass

		return self.keyword_search(query, top_k=top_k)

	def summarize_for_role(self, domain: str, role: str) -> str:
		query = f"{domain} {role} overview" if role else domain
		materials = self.retrieve_materials(query)
		context = "\n\n".join([m.get("text", "") for m in materials]) or "No relevant local documentation found."

		prompt = [
			SystemMessage(content="You are an assistant that summarizes project onboarding information for a user with a specific role."),
			HumanMessage(content=f"Domain: {domain}\nRole: {role}\n\nHere are the materials:\n{context}\n\nPlease write a 5-bullet summary tailored to the role, and suggested first 3 tasks to get started.")
		]
		resp = self.llm.generate(messages=prompt)
		try:
			text = resp.generations[0][0].text
		except Exception:
			text = getattr(resp, "text", str(resp))
		return text

	def provide_contacts_fallback(self, domain: str) -> List[Dict]:
		contacts_path = os.path.join(self.docs_dir, "contacts.json")
		if os.path.exists(contacts_path):
			try:
				with open(contacts_path, "r", encoding="utf-8") as f:
					data = json.load(f)
					matches = [c for c in data if domain.lower() in ",".join(c.get("domains", [])).lower()]
					return matches or data[:3]
			except Exception:
				pass

		return [{"name": "Team Lead (unknown)", "role": "Team Owner", "email": "contact@company.example"}]

	def generate_quiz(self, domain: str, role: str, n_questions: int = 3) -> List[Dict]:
		materials = self.retrieve_materials(f"{domain} {role} overview")
		context = "\n\n".join([m.get("text", "") for m in materials]) or "No documentation available."
		prompt = [
			SystemMessage(content="You are a helpful assistant that creates short quizzes (multiple choice) to assess onboarding knowledge."),
			HumanMessage(content=f"Based on the following materials, produce {n_questions} multiple-choice questions. Each question should have 3 options and indicate the correct option number. Materials:\n{context}")
		]
		resp = self.llm.generate(messages=prompt)
		try:
			text = resp.generations[0][0].text
		except Exception:
			text = getattr(resp, "text", str(resp))

		questions = []
		q_blocks = [b.strip() for b in text.split('\n\n') if b.strip()][:n_questions]
		for b in q_blocks:
			lines = [l.strip() for l in b.splitlines() if l.strip()]
			if not lines:
				continue
			q_text = lines[0]
			options = [l for l in lines[1:4]] if len(lines) >= 4 else lines[1:]
			answer = None
			for l in lines[4:8]:
				if l.lower().startswith("answer") or l.lower().startswith("correct"):
					answer = l
					break
			questions.append({"question": q_text, "options": options, "answer": answer})
		return questions

	def run_interactive(self):
		print("Welcome to the Agentic Onboarding Assistant.")
		domain = input("Which domain or product area are you joining? ").strip()
		role = input("What's your role? (e.g., backend engineer, product manager) ").strip()

		print("\nLoading local documents and building retriever (if available)...")
		self.load_local_documents()
		self.build_retriever()

		print("\nGenerating role-specific summary...")
		summary = self.summarize_for_role(domain, role)
		print("\n--- Summary ---\n")
		print(summary)

		while True:
			action = input("\nWhat would you like next? [materials / contact / quiz / exit]: ").strip().lower()
			if action in ("exit", "quit"):
				print("Goodbye!")
				break
			elif action == "materials":
				q = input("Enter keyword, feature id, or PBI to search for: ").strip()
				results = self.retrieve_materials(q)
				if not results:
					print("No materials found. Here are contacts to ask:")
					contacts = self.provide_contacts_fallback(domain)
					for c in contacts:
						print(f"- {c.get('name')} ({c.get('role')}): {c.get('email')}")
				else:
					print(f"Found {len(results)} materials:")
					for r in results:
						src = r.get("source") or r.get("source")
						txt = r.get("text")
						print(f"--- {src} ---\n{txt[:800]}\n")
			elif action == "contact":
				contacts = self.provide_contacts_fallback(domain)
				for c in contacts:
					print(f"- {c.get('name')} ({c.get('role')}): {c.get('email')}")
			elif action == "quiz":
				questions = self.generate_quiz(domain, role, n_questions=3)
				if not questions:
					print("Could not generate quiz (no materials).")
					continue
				score = 0
				for i, q in enumerate(questions, 1):
					print(f"\nQ{i}: {q.get('question')}")
					for idx, opt in enumerate(q.get('options', []), 1):
						print(f"  {idx}. {opt}")
					ans = input("Your answer (number): ").strip()
					correct = q.get('answer')
					if correct and ans and correct.strip().endswith(ans):
						print("Correct!")
						score += 1
					else:
						print(f"Recorded answer: {ans}. Expected: {correct}")
				print(f"\nQuiz complete. Score: {score}/{len(questions)}")
			else:
				print("Unknown action. Choose: materials, contact, quiz, exit.")


def main():
	parser = argparse.ArgumentParser(description="Agentic onboarding assistant (LangChain starter)")
	parser.add_argument("--docs", default="docs", help="Path to local docs directory (default: ./docs)")
	parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name to use (default: gpt-4o-mini)")
	args = parser.parse_args()

	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		print("ERROR: OPENAI_API_KEY not set. Set it in your environment and rerun.")
		return

	agent = OnboardingAgent(model_name=args.model, docs_dir=args.docs)
	agent.run_interactive()


if __name__ == "__main__":
	main()

