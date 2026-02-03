from rag_core import RagEngine

def main():
    engine = RagEngine()
    
    test_repo = r"C:\Users\sohamm\Desktop\dev code helper\test_project"
    
    print(f"Testing ingestion on: {test_repo}")
    
    stats = engine.ingest_codebase(test_repo)
    
    print(f"\n📊 Ingestion Stats:")
    print(f"   Files processed: {stats['files_processed']}")
    print(f"   Chunks created: {stats['chunks_created']}")
    print(f"   Duration: {stats['duration_seconds']}s")
    
    # Ask a question
    question = "What does the add function do?"
    print(f"\n❓ Asking: {question}")
    
    response = engine.ask_question(question)
    
    print("\n🤖 Answer:")
    print(response["answer"])
    
    print("\n📄 Sources:")
    for src in response["sources"]:
        print(f"- {src['file']} (Score: {src['relevance_score']:.2f})")

if __name__ == "__main__":
    main()