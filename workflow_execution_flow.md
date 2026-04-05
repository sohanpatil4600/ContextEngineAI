# Workflow Execution Flow

**Example Scenario**: The user has uploaded an academic PDF about "Neural Networks". In the Streamlit web interface, the user types: *"How do Neural Networks compare to quantum computing based on my documents and the latest web results?"*

Here is the exact journey of that query through the system:

### 1. User Input via Web UI (`app.py`)
1. The user inputs their message into Streamlit's chat box.
2. The Streamlit script checks if the system is initialized and whether the file has finished processing with TensorLake/Milvus.
3. Streamlit then invokes: `st.session_state.assistant.query(query)`.
4. The `StreamlitResearchAssistant.query` function forwards that request to the CrewAI workflow: `self.flow.kickoff(inputs={"query": user_query, "user_id": ..., "thread_id": ...})`.

### 2. State Initialization (`src/workflows/flow.py`)
5. The `kickoff()` process launches `ResearchAssistantFlow`.
6. Behind the scenes, CrewAI spins up a shared state object, mapping `ResearchAssistantState` to the active session parameters. 

### 3. Step 1: Pre-processing & Memory (`@start() process_query`)
7. **Flow Entry:** The first function `process_query()` executes instantly.
8. It takes the user's raw string: *"How do Neural Networks compare to quantum computing..."* and passes it to the internal `_summarize_for_memory()` safety check to ensure it isn't thousands of characters long.
9. **Memory Persistence:** The string is dispatched specifically to `src/memory/memory.py` leveraging `self.memory_layer.save_user_message()`, beaming the interaction log out to your active **Zep Cloud** instance.
10. The function yields the state `{'status': 'processing'}` to push the script forward.

### 4. Step 2: Parallel Context Gathering (`@listen() gather_context_from_all_sources`)
11. The script moves to `gather_context_from_all_sources()`. It creates **four explicit tasks** (from `src/workflows/tasks.py`), assigning one task to each of your data-finding agents.
12. It packages these 4 agents and 4 tasks into a mini `Crew()` run: `context_crew.kickoff()`.
13. **Parallel Execution**: All four LLM Agents begin executing concurrently using their given tools:
    *   **RAG Agent (`src/tools/rag_tool.py`)**: Submits the query against the Milvus Database instance (`rag_pipeline_retriever`). It finds the exact blocks mapping semantic chunks corresponding to "Neural Networks" from your PDF and returns them as a structured string.
    *   **Memory Agent (`src/tools/memory_tool.py`)**: Pings Zep Cloud. Retrieves past interactions checking if "Quantum Computing" was brought up five minutes ago and grabs that timeline context.
    *   **Web Search Agent (`src/tools/web_search_tool.py`)**: Uses the Firecrawl API to blindly scrape the live internet, scraping 3 real URLs for brand-new developments relating to Neural Networks versus Quantum Computing.
    *   **ArXiv Agent (`src/tools/arxiv_tool.py`)**: Hits the public ArXiv API gathering metadata and abstracts of published academic papers tied mathematically to the query prompt.
14. **Collation:** Once all 4 LLMs return strings with citations/web-links, `flow.py` attempts to run `self._parse_agent_result()` to force those raw text blocks into uniform JSON objects. 
15. This function returns the massive dictionary list called `"context_sources"` holding everything found into the active Flow state.

### 5. Step 3: Context Evaluation (`@listen() evaluate_context_relevance`)
16. The workflow triggers the `evaluate_context_relevance` phase, receiving the fat `context_sources` payload.
17. It spins up a fresh task calling out strictly the solitary `evaluator_agent`.
18. The Evaluator Agent uses OpenAI's GPT strictly looking at the inputs from the 4 previous agents. It evaluates "Are these ArXiV links actually talking about this question? Did the PDF mention quantum mechanics?"
19. It acts as a stringent bouncer. It generates a Pydantic-shaped schema mapping (`ContextEvaluationResult`), throwing away data deemed 'useless', assigning dynamic confidence scores (e.g., 0.95 for RAG, 0.40 for Memory, 0.90 for Web Search), and provides its "reasoning" on why it filtered it out.
20. The pruned, highly verified payload is saved into the state as `"filtered_context"` and passed downstream.

### 6. Step 4: Final Generation (`@listen() synthesize_final_response`)
21. The sequence invokes its final method: `synthesize_final_response`.
22. Passing forward only the heavily refined `"filtered_context"`, it gives a strict task payload to the final component: the `synthesizer_agent`.
23. The Synthesizer is instructed to write a single human-readable markdown response pulling paragraphs contextually. It embeds direct citation coordinates using the metadata inherited from Step 2.
24. **Memory Closeout:** The newly completed finalized answer text is piped immediately back to `_summarize_for_memory()`, recording the assistant's final thought to **Zep Cloud** inside the thread registry securely.
25. The Flow resolves, outputting `{"final_response": "The markdown generated answering the question...", "status": "completed"}` back to the web frontend execution loop.

### 7. View Update (`app.py`)
26. Execution snaps back to Streamlit script inside `app.py`.
27. Streamlit unpacks `result['final_response']` and forcibly draws a brand-new markdown bubble on the UI chat screen showing the formatted answer.
28. Right alongside it, Streamlit runs `display_citations_dropdown(response)`. This scrapes the payload for the confidence scores from the Evaluator Agent and generates the drop-down accordion menus so that the User can interactively inspect exactly what PDF chunks or Web links were proven useful for constructing the final block.
