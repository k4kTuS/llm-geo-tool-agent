from openai import OpenAI

client = OpenAI()
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import torch

from paths import PROJECT_ROOT

@st.cache_resource
def load_emb_model():
    return SentenceTransformer("intfloat/multilingual-e5-small")

base_prompt = """
Given the user's query, analyze and refine it by splitting it into independent topics that may be related to the list of given tools.
Do not include tools themselves, only the information from the User's prompt.
Ensure that each topic is self-contained and meaningful, so embeddings can be generated later.  
Preserve the original intent while making the topics clearer and more structured.
Split these topics by '%' symbol
"""

def refine_user_prompt_openai(users_prompt: str, 
                              base_prompt: str,
                              tool_list: str,
                              max_target_length: int = 512, 
                              temperature: float = 0.2) -> str:
    """
    Processes the user's query by splitting it into semantically similar topics.

    Args:
        users_prompt (str): The user's input query .
        base_prompt (str): The base instruction for refining and splitting the query.
        tool_list (str): available list of tools.
        max_target_length (int, optional): The maximum length of the generated response. Defaults to 512.
        temperature (float, optional): Controls randomness in response generation. Defaults to 0.2.

    Returns:
        str: The processed query, split into semantically meaningful topics.
    """
    try:
        response = client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that splits a user's query into semantically similar topics."},
            {"role": "user", "content": f"{base_prompt}\n\nUser's prompt:\n{users_prompt}\n\n{', '.join(tool_list)}"}
        ],
        temperature=temperature,
        max_tokens=max_target_length,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_relevant_tools(output, tool_list, embedding_model, threshold=0.3):
    """
    Finds tools relevant to each output based on cosine similarity and a threshold.

    Args:
        output (str): A single string containing multiple outputs separated by '%'.
        tool_list (list[str]): List of tool names or descriptions.
        embedding_model: The embedding model used to generate embeddings.
        threshold (float, optional): Cosine similarity threshold to filter relevant tools. Defaults to 0.3.

    Returns:
        set: A set of unique relevant tools.
    """

    # Ensure output is properly split
    output_list = output.split(' %') if isinstance(output, str) else output

    # Compute embeddings
    output_embeddings = embedding_model.encode(output_list, convert_to_tensor=True)
    tool_embeddings = embedding_model.encode(tool_list, convert_to_tensor=True)

    # Compute cosine similarity
    similarity_matrix = util.cos_sim(output_embeddings, tool_embeddings)

    # Collect relevant tools into a set
    relevant_tools = set()
    for i in range(len(output_list)):
        mask = similarity_matrix[i] >= threshold
        selected_indices = torch.nonzero(mask, as_tuple=True)[0]  # Get indices of tools above threshold

        # Add tools to the set
        relevant_tools.update({tool_list[j] for j in selected_indices})

    return list(relevant_tools)

def filter_tools(tool_names: list[str], query: str = None):
    output = refine_user_prompt_openai(
        query,
        base_prompt,
        tool_names,
    )
    print("Query transformation:", output)
    embedding_model = load_emb_model()
    filtered_tools = get_relevant_tools(output, tool_names, embedding_model, threshold=0.8)
    return filtered_tools