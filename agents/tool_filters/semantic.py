from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import torch

from agents.tool_filters.base import BaseToolFilter
from agents.prompts import QUERY_REFINEMENT

client = OpenAI()

@st.cache_resource
def load_emb_model():
    return SentenceTransformer("intfloat/multilingual-e5-small")

class SemanticToolFilter(BaseToolFilter):
    def filter(self, tools, context):
        messages =  context.get("messages")
        query = messages[-1].content if messages else None

        if query is None:
            raise ValueError("No query provided for filtering tools")

        tool_names = [tool.name for tool in tools]
        names_to_tools = {tool.name: tool for tool in tools}
        filtered_tools = self._filter_tools(tool_names, query)
        filtered = [names_to_tools[name] for name in filtered_tools if name in names_to_tools]
        return filtered

    def _refine_user_prompt_openai(self,
                                users_prompt: str, 
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

    def _get_relevant_tools(self, output, tool_list, embedding_model, threshold=0.3):
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

    def _filter_tools(self, tool_names: list[str], query: str = None):
        output = self._refine_user_prompt_openai(
            query,
            QUERY_REFINEMENT,
            tool_names,
        )
        print("Query transformation:", output)
        embedding_model = load_emb_model()
        filtered_tools = self._get_relevant_tools(output, tool_names, embedding_model, threshold=0.8)
        return filtered_tools