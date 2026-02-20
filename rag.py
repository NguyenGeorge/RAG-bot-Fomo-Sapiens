import os
import gc
import time
import json
import random
from datetime import datetime
from operator import itemgetter

# Import Data Handling and Web Scraping tools
import numpy as np
import pandas as pd
import bs4
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, GenerationConfig
from accelerate import PartialState

# Import Database & Cloud Connectivity
import certifi
from pymongo import MongoClient
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# Import Langchain - Google tools
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Import other Langchain tools:
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFacePipeline
from langchain_community.utils.math import cosine_similarity
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

gc.collect()
torch.cuda.empty_cache()

# Initialize the Kaggle Secrets client
user_secrets = UserSecretsClient()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = user_secrets.get_secret("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "Kaggle_Practice"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

# Safely retrieve the key and set it as an environment variable
os.environ["LANGCHAIN_API_KEY"] = user_secrets.get_secret("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GEMINI")
os.environ["HF_API_KEY"] = user_secrets.get_secret("HF_TOKEN")

# Set other necessary variables for LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Kaggle_Practice"

# Identify your requests (you can put any name here)
os.environ["USER_AGENT"] = "MyKaggleRAGProject"

print("LangChain environment variables and APIs set successfully!")

# Prompt Templates before Routing
vectorized_database_template = """### ROLE: Group Historian (MongoDB Router)
You are the memory of this community. 
- TRIGGER: Use this route if the [USER QUERY] asks for specific past events, decisions, names of members, or "what did we talk about regarding X."
- SOURCE: Strictly use [COMMUNITY DATABASE] to provide factual citations from chat history.
- TONE: Informative but integrated, like a member with a perfect memory.
"""

fine_tuned_model_template = """ ### ROLE: The Old Friend (Fine-Tuned Persona)
You are a long-time member who embodies the group's "vibe."
- TRIGGER: Use this route for casual banter, emotional venting, or general "How are you" style talk where facts don't matter.
- STYLE: Low-stakes, high-emotion, and slang-heavy (matching the group). 
- CONSTRAINT: Do not act like an AI or a lecturer. If the user isn't asking for a fact, just be a friend.
"""

mixed_template = """ ### ROLE: Group Research Assistant (External Search)
You are the bridge between this group and the outside world.
- TRIGGER: Use this route if the [USER QUERY] asks for definitions, technical explanations, news, or topics that haven't been discussed in the group.
- SOURCE: Search the Internet/Gemini to bring fresh knowledge into the chat.
- TONE: Helpful and smart, but translate complex info into a style the group understands.
"""

# Set up Gemini LLM Connection
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", 
                                      google_api_key=os.environ["GOOGLE_API_KEY"] , 
                                      temperature=0.7)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=3072
)

prompt_templates = [vectorized_database_template, fine_tuned_model_template, mixed_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Set up Hugging Face Connection
login(token=os.environ["HF_API_KEY"])
print("Connected to Hugging Face!")

def initialize_hf_llm(model_id):
    """
    Initializes a quantized Llama-3 model and returns a LangChain-compatible 
    HuggingFacePipeline for local inference.
    """
    
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Quantization Configuration (4-bit for VRAM efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_quant_type="nf4",             
        bnb_4bit_use_double_quant=True         
    )

    # Generation Configuration
    gen_config = GenerationConfig(
        max_new_tokens=4096,
        temperature=0.88,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id, 
        top_p=0.7,
        top_k=50,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    )
    distributed_state = PartialState()

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": distributed_state.device},
        #max_memory={0: "10GiB", 1: "14GiB"},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.generation_config = gen_config

    # Create Pipeline
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        return_full_text=False,
        batch_size=1
    )
    
    hf_llm = HuggingFacePipeline(pipeline=pipe)
    
    print(f"Setup Hugging Face model {model_id} - done!")
    return hf_llm

# Set up MongoDB Connection
def initialize_vector_store(client, db_name, collection_name, embeddings):
    """
    Wraps an existing MongoDB collection into a LangChain VectorStore.
    """
    db = client[db_name]
    collection = db[collection_name]
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings, 
        index_name="vector_index", 
        relevance_score_fn="cosine"
    )
    
    print("Vector Store wrapper initialized.")
    return vector_store

def generate_multi_queries(query, model):
    multi_query_prompt = f"""
    You are an expert at query expansion for crypto community chats. 
    Your goal is to take the user's input and generate 3 DIFFERENT perspectives to search a database. Prioritize Vietnamese.

    User Input: "{query}"

    Instructions:
    1. Version 1 (Technical/Literal): Focus on the core entities, links, or technical terms.
    2. Version 2 (Vietnamese Slang/Vibe): Rewrite the query as if a degen in a crypto group is asking it.
    3. Version 3 (Contextual/Broad): Expand on what this link or question might be about.

    Output ONLY a JSON list of 3 distinct strings. No numbering, no extra text.
    
    """
    
    response = model.invoke(multi_query_prompt)
    
    try:
        # Clean up markdown if Gemini includes it
        raw_content = response.content.replace("```json", "").replace("```", "").strip()
        queries = json.loads(raw_content)
        
        # Safety check: if they are identical, force a manual variation
        if len(set(queries)) == 1:
            return [query, f"check thông tin {query}", f"vibe của {query} thế nào"]
            
        return queries
    except Exception as e:
        print(f"Multi-query parsing error: {e}")
        return [query, query, query]

def prompt_router(queries_list, prompt_embeddings):
    """
    Routes based on the 'average intent' of multiple query versions.
    queries_list: The output from generate_multi_queries
    prompt_embeddings: Pre-calculated embeddings of your 3 templates
    """
    # Embed all 3 versions
    query_embeddings = embeddings.embed_documents(queries_list)
    
    # Calculate similarity for each query version against each template
    sim_matrix = cosine_similarity(query_embeddings, prompt_embeddings)
    
    # Average the scores across all query versions
    avg_scores = np.mean(sim_matrix, axis=0)
    
    route_names = ["mongodb", "fine_tuned", "mixed"]
    best_route = route_names[np.argmax(avg_scores)]
    
    print(f"Router Decision: {best_route} | Confidence: {avg_scores.tolist()}")
    return best_route

def vectorized_mongodb_handler(input_data, vector_store):
    if isinstance(input_data, list):
        queries = input_data
    else:
        queries = input_data.get("query", [])
        if isinstance(queries, str):
            queries = [queries]

    unique_queries = list(dict.fromkeys(queries))
    original_query = unique_queries[0]
    
    threshold = 0.8
    all_candidate_docs = []
    seen_ids = set()

    # Generate vector for the original query (used for final scoring)
    original_vector = embeddings.embed_query(original_query, output_dimensionality=768)

    # Search (Only for Unique Queries)
    for q_text in unique_queries:
        q_vector = embeddings.embed_query(q_text, output_dimensionality=768)
        
        # Use MMR to increase the scope of searching
        docs = vector_store.max_marginal_relevance_search_by_vector(
            q_vector,
            k=15,
            fetch_k=40,
            lambda_mult=0.9 
        )
        
        for doc in docs:
            # Deduplicate by MongoDB ID
            doc_id = str(doc.metadata.get("_id"))
            if doc_id not in seen_ids:
                all_candidate_docs.append(doc)
                seen_ids.add(doc_id)

    if not all_candidate_docs:
        return {"prompt": f"No history found for: {original_query}"}

    # Batch Ranking based on the initial MMR Search Pool
    doc_vectors = [doc.metadata.get("embedding") for doc in all_candidate_docs if "embedding" in doc.metadata]
    
    if not doc_vectors:
        # If no embeddings found in metadata, fallback to MMR top results
        docs = all_candidate_docs[:6]
    else:
        # Convert vector to matrix
        X = np.array(original_vector).reshape(1, -1) 
        Y = np.array(doc_vectors)
        
        scores = cosine_similarity(X, Y)[0] 

        final_docs = []
        for i, score in enumerate(scores):
            print(f"Index {i} and score: {score:.4f}")
            if score >= threshold:
                doc = all_candidate_docs[i]
                doc.metadata["score"] = score
                final_docs.append(doc)
        
        docs = sorted(final_docs, key=lambda x: x.metadata.get("score", 0), reverse=True)[:6]
    if not docs:
        docs = all_candidate_docs[:3]

    print("printing docs...", docs)

    # Format for the Prompt
    style_list = [f"{d.metadata.get('from', 'Member')}: {d.page_content.strip()}" for d in docs]
    chat_logs_str = "\n".join(style_list)
    vectorized_database_prompt = """### ROLE
    You are a core member of this community. Your goal is to answer the [USER QUERY] by using the facts found in the [COMMUNITY DATABASE].
    
    ### COMMUNITY DATABASE (Your Knowledge Source)
    The following messages contain the facts, history, and information you need to answer. 
    <database_facts>
    {chat_logs_str}
    </database_facts>
    
    ### STYLE GUIDELINES (How to speak)
    Analyze the vocabulary, slang, and sentence structure in the messages above. 
    - Do NOT sound like an AI assistant.
    - Use the community's shorthand and "vibe."
    - If the database facts don't contain the answer, say so in the community's style—don't make things up.
    
    ### USER QUERY
    {original_query}
    
    ### YOUR RESPONSE
    """
    return {"prompt": vectorized_database_prompt}

def fine_tuned_model_handler(input_data, vector_store):
    """
    Inputs 3 queries, finds the highest score match, extracts metadata, 
    and returns a single formatted prompt.
    """
    queries = input_data if isinstance(input_data, list) else input_data.get("query", [])
    
    best_score = -1
    winning_doc = None
    winning_query = None

    # Ranking Loop
    for query in queries:
        query_vector = embeddings.embed_query(query, output_dimensionality=768)
        results = vector_store._similarity_search_with_score(query_vector, k=1)
        if results:
            # Unpack the first (and only, since k=1) result
            doc, score = results[0]
            
            try:
                score = float(score)
                
                # MongoDB Atlas (Cosine) usually gives scores where higher is better
                if score > best_score: 
                    best_score = score
                    winning_doc = doc
                    winning_query = query
            except (ValueError, TypeError):
                continue

    # Safety Check: If no results found at all
    if not winning_doc:
        return {"prompt": "No relevant history found for these queries."}
    
    # Metadata Extraction - from vectorized MongoDB
    users = winning_doc.metadata.get("metadata", {}).get("u", ["Satoshi"])
    target_persona = random.choice(users) if users else "Satoshi"
    raw_date = winning_doc.metadata.get("metadata", {}).get("date", datetime.now())
    formatted_time = raw_date.strftime("%Y-%m-%d %H:%M") if hasattr(raw_date, 'strftime') else str(raw_date)

    # Final Prompt Design
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are mimicking the chat style of {target_persona}. "
        f"Context: {winning_doc.page_content}<|eot_id|>"
        f"<|start_header_id|>User - Gemini<|end_header_id|>\n\n"
        f"Datetime - [{formatted_time}] User's Message - #####USER#####{winning_query}<|eot_id|>"
        f"<|start_header_id|>Assistant - {target_persona}<|end_header_id|>\n\n"
        f"Datetime - [{formatted_time}] Assistant's Message -#####RESPONSE#####"
    ).strip()

    return {"prompt": formatted_prompt}

def mixed_handler(input_data):
    """
    Search google, summarize the context, then force them to speak in the group style.
    """
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=os.environ["GOOGLE_API_KEY"])
    if isinstance(input_data, list):
        queries = input_data
    elif isinstance(input_data, dict):
        queries = input_data.get("query", [])
    else:
        queries = [str(input_data)]
    print(queries)
    unique_queries = list(set(queries)) if queries else ["Hello"]
    user_input = unique_queries[0]
    raw_results = []
    print("Slowing down for API quota safety...")
    time.sleep(2)
    research_template = ChatPromptTemplate.from_template(
        "You are a Research Assistant. Take these 3 queries: {queries}. "
        "Search your internal knowledge for facts related to the topics, keywords, "
        "and names mentioned. Return a concise 300-character summary of the key facts."
    )
    
    # Execute the Research Chain
    research_chain = research_template | model | StrOutputParser()
    
    # We pass the list of queries directly to Gemini
    try:
        summary_context = research_chain.invoke({"queries": ", ".join(unique_queries)})
    except Exception as e:
        print(f"API Error: {e}. Using fallback context.")
        summary_context = "No additional facts found."

    # Format the final Llama-3 Persona Prompt
    target_persona = "Gemini"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    formatted_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are mimicking the chat style of {target_persona}. "
        f"Use the following facts to inform your response: {summary_context}<|eot_id|>"
        f"<|start_header_id|>User- Jarvis<|end_header_id|>\n\n"
        f"Datetime - [{current_time}] User's Message - #####USER#####{user_input}<|eot_id|>"
        f"<|start_header_id|>Assistant - {target_persona}<|end_header_id|>\n\n"
        f"Datetime - [{current_time}] Assistant's Message -#####RESPONSE#####"
    ).strip()
    
    return {"prompt": formatted_prompt}
    
def main_orchestrator(input_data, vector_store, model):
    # Determine the route
    query = input_data.get("query")
    expanded_queries = generate_multi_queries(query, model=gemini_model)
    print(expanded_queries)
    route = prompt_router(expanded_queries, prompt_embeddings)
    prompt_content = ""
    
    # Execute the appropriate handler
    if route == "mongodb":
        print("Using MongoDB\n")
        prompt_content = vectorized_mongodb_handler(input_data=expanded_queries, vector_store=vector_store)["prompt"]
    elif route == "fine_tuned":
        print("Using Hugging Face fine-tuned model\n")
        prompt_content = fine_tuned_model_handler(input_data=expanded_queries, vector_store=vector_store)["prompt"]
    else:
        print("Using mixed template - Gemini search + Fine-tuned model\n")
        prompt_content = mixed_handler(input_data=expanded_queries)["prompt"]

    print("printing prompt....", prompt_content)
    return {
        "route": route,            # Crucial for RunnableBranch
        "prompt": prompt_content,  # The actual text for the LLM
        "query": input_data.get("query")
    }

def clear_all_gpu_cache():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        # Set the current device to the specific GPU
        torch.cuda.set_device(i)
        # Clear the cache for that specific device
        torch.cuda.empty_cache()
        print(f"Cleared cache on GPU {i}")

def main():
    hf_model_id = "GeorgeNguyen/llama-3-groupchat-final"
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    extract_str_prompt = RunnableLambda(lambda x: x["prompt"] if isinstance(x, dict) else str(x))

    hf_llm = initialize_hf_llm(model_id=hf_model_id)
    mongodb_uri = user_secrets.get_secret("Mongodb_Uri")
    client = MongoClient(mongodb_uri, tlsCAFile=certifi.where())

    vector_store = initialize_vector_store(client=client, 
                                           db_name="TelegramData", 
                                           collection_name="messages", 
                                           embeddings=embeddings)

    branch = RunnableBranch(
        # IF route is fine_tuned -> Use the prompt key for the chat_model
        (
            lambda x: x["route"] in ["fine_tuned", "mixed"],
            extract_str_prompt | hf_llm
            
        ),
        # ELSE (Default) -> Use Gemini with the prompt key
        extract_str_prompt | ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    )

    chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(lambda x: main_orchestrator(x, vector_store, model))
        | branch
        | extract_str_prompt 
        | ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        | StrOutputParser()
    )
    
    print("Ready for chat!")
    
    question = "blockchain là gì?"
    print(question + "\n")
    with torch.no_grad():
        response = chain.invoke(question)
        print(response)
    
    # Clear the cache
    clear_all_gpu_cache()
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
