import csv, hashlib, html, json, os, re
from dataclasses import asdict, dataclass
from pathlib import Path
import chromadb, numpy as np

for line in Path(".env").read_text(encoding="utf-8").splitlines() if Path(".env").exists() else []:
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1); os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
def env(k,d): return os.getenv(k,d)
@dataclass(frozen=True)
class Settings:
    data_dir: Path=Path(env("DATA_DIR","data/raw")); index_dir: Path=Path(env("INDEX_DIR","chroma_db"))
    chunk_size:int=int(env("CHUNK_SIZE","900")); chunk_overlap:int=int(env("CHUNK_OVERLAP","150")); parent_chunk_size:int=int(env("PARENT_CHUNK_SIZE","1800")); parent_chunk_overlap:int=int(env("PARENT_CHUNK_OVERLAP","250"))
    top_k:int=int(env("TOP_K","5")); dimensions:int=int(env("EMBEDDING_DIMENSIONS","768")); collection:str=env("CHROMA_COLLECTION","rag_documents")
    llm_provider:str=env("LLM_PROVIDER","groq").lower(); llm_model:str=env("LLM_MODEL","openai/gpt-oss-20b"); temperature:float=float(env("TEMPERATURE","0.1"))
@dataclass
class Chunk: text:str; search_text:str; source:str; filename:str; document_type:str; page:int|None; chunk_id:int; parent_id:int
def clean(t): return re.sub(r"\s+"," ",t.replace("\x00","")).strip()
def split(t,size,overlap):
    for start in range(0,len(t),size-overlap):
        piece=t[start:start+size]
        if piece: yield piece
        if start+size>=len(t): break
def load(path):
    if path.suffix.lower()==".pdf":
        from pypdf import PdfReader
        for n,p in enumerate(PdfReader(str(path)).pages,1): yield p.extract_text() or "",n
    elif path.suffix.lower()==".csv":
        with path.open(encoding="utf-8-sig",newline="") as f: yield "\n".join(" | ".join(r) for r in csv.reader(f)),None
    elif path.suffix.lower()==".docx":
        from docx import Document
        yield "\n".join(p.text for p in Document(path).paragraphs),None
    else: yield path.read_text(encoding="utf-8",errors="replace"),None
def embed(texts,d):
    a=np.zeros((len(texts),d),dtype=np.float32)
    for i,t in enumerate(texts):
        for token in re.findall(r"\w+",t.lower()): a[i,int(hashlib.sha256(token.encode()).hexdigest(),16)%d]+=1
    return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
class RAGPipeline:
 def __init__(self): self.s=Settings(); self.client=chromadb.PersistentClient(path=str(self.s.index_dir)); self.chunks=[]; self.collection=None
 def rebuild(self):
    self.chunks=[]
    for path in self.s.data_dir.rglob("*"):
      if path.is_file():
       for text,page in load(path):
        for parent in split(clean(text),self.s.parent_chunk_size,self.s.parent_chunk_overlap):
         pid=len(self.chunks)
         for child in split(parent,self.s.chunk_size,self.s.chunk_overlap): self.chunks.append(Chunk(parent,child,str(path),path.name,path.suffix[1:].upper(),page,len(self.chunks),pid))
    if not self.chunks: raise RuntimeError("Put documents in data/raw, then rebuild.")
    try:self.client.delete_collection(self.s.collection)
    except Exception:pass
    self.collection=self.client.create_collection(self.s.collection,metadata={"hnsw:space":"cosine"}); vectors=embed([c.search_text for c in self.chunks],self.s.dimensions)
    for i in range(0,len(self.chunks),100):
      b=self.chunks[i:i+100]; self.collection.add(ids=[str(c.chunk_id) for c in b],documents=[c.search_text for c in b],embeddings=vectors[i:i+100].tolist())
    Path("data/chunks").mkdir(parents=True,exist_ok=True); Path("data/chunks/chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]),encoding="utf-8")
 def load(self):
    self.chunks=[Chunk(**x) for x in json.loads(Path("data/chunks/chunks.json").read_text(encoding="utf-8"))]; self.collection=self.client.get_collection(self.s.collection)
 def answer(self,q):
    r=self.collection.query(query_embeddings=embed([q],self.s.dimensions).tolist(),n_results=min(20,self.collection.count()),include=["distances"]); seen=set(); found=[]
    for ident in r["ids"][0]:
      c=self.chunks[int(ident)]
      if c.parent_id not in seen: found.append(c); seen.add(c.parent_id)
      if len(found)==self.s.top_k:break
    context="\n\n".join(f"[{i+1}] {c.text}" for i,c in enumerate(found))
    answer=self._generate(q, context, found)
    chunks="\n\n".join(f"--- Chunk {i+1}: {c.filename}, page {c.page} ---\n{c.text}" for i,c in enumerate(found))
    return f"Answer:\n{answer}\n\nTop {len(found)} retrieved chunks:\n{chunks}"
 def _generate(self, question, context, found):
    prompt=f"Answer the question exactly and only from the context. If the answer is not present, say so. Cite supporting chunk numbers like [1].\n\nContext:\n{context}\n\nQuestion: {question}"
    try:
      if self.s.llm_provider=="groq" and os.getenv("GROQ_API_KEY"):
       from groq import Groq
       return Groq().chat.completions.create(model=self.s.llm_model,temperature=self.s.temperature,messages=[{"role":"system","content":"You are a precise, source-grounded RAG assistant."},{"role":"user","content":prompt}]).choices[0].message.content
      if self.s.llm_provider=="openai" and os.getenv("OPENAI_API_KEY"):
       from openai import OpenAI
       return OpenAI().chat.completions.create(model=self.s.llm_model,temperature=self.s.temperature,messages=[{"role":"system","content":"You are a precise, source-grounded RAG assistant."},{"role":"user","content":prompt}]).choices[0].message.content
    except Exception: pass
    # Local fallback: return the best matching sentence when an LLM is unavailable.
    terms=set(re.findall(r"\w+",question.lower()))
    sentences=re.split(r"(?<=[.!?])\s+", " ".join(c.text for c in found))
    best=max(sentences, key=lambda sentence: len(terms & set(re.findall(r"\w+",sentence.lower()))), default="")
    return best or "No relevant information was found."
