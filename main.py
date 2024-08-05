from langchain.prompts import StringPromptTemplate
from typing import Callable
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
import os
import json
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
import pandas as pd
from langchain.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq

os.environ["GROQ_API_KEY"] = ""
url = ""
username = ""
password = ""

graph = Neo4jGraph(
    url=url, 
    username=username, 
    password=password
)

file_path = 'amazon_product_kg.json'

def sanitize(text):
    text = str(text).replace("'","").replace('"','').replace('{','').replace('}', '')
    return text

def check_data_exists():
    query = "MATCH (n) RETURN COUNT(n) as count"
    result = graph.query(query)
    return result[0]['count'] > 0

def insert_data():
    with open(file_path, 'r') as file:
        jsonData = json.load(file)

    entity_types = set()

    for i, obj in enumerate(jsonData, 1):
        print(f"{i}. {obj['product_id']} -{obj['relationship']}-> {obj['entity_value']}")
        query = f'''
            MERGE (product:Product {{id: {obj['product_id']}}})
            ON CREATE SET product.name = "{sanitize(obj['product'])}", 
                           product.title = "{sanitize(obj['TITLE'])}", 
                           product.bullet_points = "{sanitize(obj['BULLET_POINTS'])}", 
                           product.size = {sanitize(obj['PRODUCT_LENGTH'])}

            MERGE (entity:{obj['entity_type']} {{value: "{sanitize(obj['entity_value'])}"}})

            MERGE (product)-[:{obj['relationship']}]->(entity)
            '''
        graph.query(query)
        entity_types.add(obj['entity_type'])

    print("Data insertion completed.")
    return list(entity_types)

# Check if data exists before inserting
if not check_data_exists():
    print("Database is empty. Inserting data...")
    entities_list = insert_data()
else:
    print("Data already exists in the database. Skipping insertion.")
    # Fetch existing entity types from the database
    query = "MATCH (n) WHERE NOT n:Product RETURN DISTINCT labels(n) as entity_types"
    result = graph.query(query)
    entities_list = [label[0] for sublist in result for label in sublist['entity_types']]

# Use the model name directly instead of creating a separate embeddings_model object
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_index(index_name, node_label, text_properties):
    return Neo4jVector.from_existing_graph(
        embeddings,
        url=url,
        username=username,
        password=password,
        index_name=index_name,
        node_label=node_label,
        text_node_properties=text_properties,
        embedding_node_property='embedding',
    )

# Create vector index for products
product_index = create_vector_index('products', 'Product', ['name', 'title'])

# Create vector indices for entities
for entity_type in entities_list:
    create_vector_index(entity_type, entity_type, ['value'])

print("Vector indices creation completed.")
print(f"Created indices for the following entity types: {', '.join(entities_list)}")

from langchain.chains import GraphCypherQAChain
from langchain_groq import ChatGroq

groq_model = ChatGroq(
    model_name="llama3-8b-8192" # You can choose a different model if needed
)

# Create the GraphCypherQAChain with Groq
chain = GraphCypherQAChain.from_llm(
    llm=groq_model,
    graph=graph,
    verbose=True,
)

entity_types = {
    "product": "Item detailed type, for example 'high waist pants', 'outdoor plant pot', 'chef kitchen knife'",
    "category": "Item category, for example 'home decoration', 'women clothing', 'office supply'",
    "characteristic": "if present, item characteristics, for example 'waterproof', 'adhesive', 'easy to use'",
    "measurement": "if present, dimensions of the item", 
    "brand": "if present, brand of the item",
    "color": "if present, color of the item",
    "age_group": "target age group for the product, one of 'babies', 'children', 'teenagers', 'adults'. If suitable for multiple age groups, pick the oldest (latter in the list)."
}

relation_types = {
    "hasCategory": "item is of this category",
    "hasCharacteristic": "item has this characteristic",
    "hasMeasurement": "item is of this measurement",
    "hasBrand": "item is of this brand",
    "hasColor": "item is of this color", 
    "isFor": "item is for this age_group"
 }

entity_relationship_match = {
    "category": "hasCategory",
    "characteristic": "hasCharacteristic",
    "measurement": "hasMeasurement", 
    "brand": "hasBrand",
    "color": "hasColor",
    "age_group": "isFor"
}

system_prompt = f'''
    You are a helpful agent designed to fetch information from a graph database. 
    
    The graph database links products to the following entity types:
    {json.dumps(entity_types)}
    
    Each link has one of the following relationships:
    {json.dumps(relation_types)}

    Depending on the user prompt, determine if it possible to answer with the graph database.
        
    The graph database can match products with multiple relationships to several entities.
    
    Example user input:
    "Which blue clothing items are suitable for adults?"
    
    There are three relationships to analyse:
    1. The mention of the blue color means we will search for a color similar to "blue"
    2. The mention of the clothing items means we will search for a category similar to "clothing"
    3. The mention of adults means we will search for an age_group similar to "adults"
    
    
    Return a json object following the following rules:
    For each relationship to analyse, add a key value pair with the key being an exact match for one of the entity types provided, and the value being the value relevant to the user query.
    
    For the example provided, the expected output would be:
    {{
        "color": "blue",
        "category": "clothing",
        "age_group": "adults"
    }}
    
    If there are no relevant entities in the user prompt, return an empty json object.
'''

client=Groq()

def define_query(prompt, model="mixtral-8x7b-32768"):
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        #print("API Response:", completion)  # Debug print
        
        # Check if 'choices' attribute exists
        if hasattr(completion, 'choices'):
            return completion.choices[0].message.content
        else:
            # If 'choices' doesn't exist, return the entire response
            return str(completion)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def create_embedding(text):
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = model.embed_documents([text])
    return embeddings[0] 




# The threshold defines how closely related words should be. Adjust the threshold to return more or less results
def query_graph(response):
    embeddingsParams = {}
    query = create_query(response)
    query_data = json.loads(response)
    for key, val in query_data.items():
        embeddingsParams[f"{key}Embedding"] = create_embedding(val)
    
    print("Generated Cypher Query:")
    print(query)
    print("\nQuery Parameters:")
    print(json.dumps(embeddingsParams, indent=2))

    result = graph.query(query, params=embeddingsParams)
    return result

def create_query(text, threshold=3):  # Increased threshold for more lenient matching
    query_data = json.loads(text)
    embeddings_data = []
    for key, val in query_data.items():
        if key != 'product':
            embeddings_data.append(f"${key}Embedding AS {key}Embedding")
    query = "WITH " + ",\n".join(e for e in embeddings_data)
    query += "\nMATCH (p:Product)\nMATCH "
    match_data = []
    for key, val in query_data.items():
        if key != 'product':
            relationship = entity_relationship_match[key]
            match_data.append(f"(p)-[:{relationship}]->({key}Var:{key})")
    query += ",\n".join(e for e in match_data)
    similarity_data = []
    for key, val in query_data.items():
        if key != 'product':
            similarity_data.append(f"""
                CASE
                    WHEN {key}Var.embedding IS NOT NULL AND size({key}Var.embedding) > 0
                    THEN reduce(s = 0, i in range(0, size({key}Var.embedding)) | 
                         s + ({key}Var.embedding[i] - ${key}Embedding[i]) ^ 2) < {threshold}
                    ELSE true  // Changed to true to include nodes without embeddings
                END
            """)
    query += "\nWHERE " + " AND ".join(e for e in similarity_data)
    query += "\nRETURN p, "
    query += ", ".join(f"{key}Var" for key in query_data if key != 'product')
    return query



# Adjust the relationships_threshold to return products that have more or less relationships in common
def query_similar_items(product_id, relationships_threshold = 3):
    
    similar_items = []
        
    # Fetching items in the same category with at least 1 other entity in common
    query_category = '''
            MATCH (p:Product {id: $product_id})-[:hasCategory]->(c:category)
            MATCH (p)-->(entity)
            WHERE NOT entity:category
            MATCH (n:Product)-[:hasCategory]->(c)
            MATCH (n)-->(commonEntity)
            WHERE commonEntity = entity AND p.id <> n.id
            RETURN DISTINCT n;
        '''
    

    result_category = graph.query(query_category, params={"product_id": int(product_id)})
    #print(f"{len(result_category)} similar items of the same category were found.")
          
    # Fetching items with at least n (= relationships_threshold) entities in common
    query_common_entities = '''
        MATCH (p:Product {id: $product_id})-->(entity),
            (n:Product)-->(entity)
            WHERE p.id <> n.id
            WITH n, COUNT(DISTINCT entity) AS commonEntities
            WHERE commonEntities >= $threshold
            RETURN n;
        '''
    result_common_entities = graph.query(query_common_entities, params={"product_id": int(product_id), "threshold": relationships_threshold})
    #print(f"{len(result_common_entities)} items with at least {relationships_threshold} things in common were found.")

    for i in result_category:
        similar_items.append({
            "id": i['n']['id'],
            "name": i['n']['name']
        })
            
    for i in result_common_entities:
        result_id = i['n']['id']
        if not any(item['id'] == result_id for item in similar_items):
            similar_items.append({
                "id": result_id,
                "name": i['n']['name']
            })
    return similar_items


#product_ids = ['1519827', '2763742']#----use this for finding similiar items

#for product_id in product_ids:
    #print(f"Similar items for product #{product_id}:\n")
    #result = query_similar_items(product_id)
    #print("\n")
    #for r in result:
        #print(f"{r['name']} ({r['id']})")
    #print("\n\n")----------------------------------use this for similair items
    
    

def query_db(params):
    matches = []
    # Querying the db
    result = query_graph(params)
    for r in result:
        product_id = r['p']['id']
        matches.append({
            "id": product_id,
            "name":r['p']['name']
        })
    return matches    




import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def similarity_search(prompt, threshold=0.3):  # Adjust threshold as needed
    matches = []
    embedding = create_embedding(prompt)
    
    # Fetch all product embeddings from the Neo4j database
    query = '''
            MATCH (p:Product)
            RETURN p.id AS id, p.name AS name, p.embedding AS embedding
            '''
    result = graph.query(query)
    
    # Convert result to a list of dictionaries
    products = [{"id": r['id'], "name": r['name'], "embedding": r['embedding']} for r in result]
    
    # Compute cosine similarity between the query embedding and each product embedding
    for product in products:
        product_embedding = np.array(product['embedding']).reshape(1, -1)
        query_embedding = np.array(embedding).reshape(1, -1)
        similarity = cosine_similarity(query_embedding, product_embedding)[0][0]
        
        # Filter products based on the similarity threshold
        if similarity > threshold:
            matches.append({
                "id": product['id'],
                "name": product['name']
            })
    
    return matches

prompt_similarity = "I'm looking for nice curtains"
print(similarity_search(prompt_similarity))





tools = [
    Tool(
        name="Query",
        func=query_db,
        description="Use this tool to find entities in the user prompt that can be used to generate queries"
    ),
    Tool(
        name="Similarity Search",
        func=similarity_search,
        description="Use this tool to perform a similarity search with the products in the database"
    )
]

tool_names = [f"{tool.name}: {tool.description}" for tool in tools]




prompt_template = '''Your goal is to find a product in the database that best matches the user prompt.
You have access to these tools:

{tools}

Use the following format:

Question: the input prompt from the user
Thought: you should always think about what to do
Action: the action to take (refer to the rules below)
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules to follow:

1. Start by using the Query tool with the prompt as parameter. If you found results, stop here.
2. If the result is an empty array, use the similarity search tool with the full initial user prompt. If you found results, stop here.
3. If you cannot still cannot find the answer with this, probe the user to provide more context on the type of product they are looking for. 

Keep in mind that we can use entities of the following types to search for products:

{entity_types}.

3. Repeat Step 1 and 2. If you found results, stop here.

4. If you cannot find the final answer, say that you cannot help with the question.

Never return results if you did not find any results in the array returned by the query tool or the similarity search tool.

If you didn't find any result, reply: "Sorry, I didn't find any suitable products."

If you found results from the database, this is your final answer, reply to the user by announcing the number of results and returning results in this format (each new result should be on a new line):

name_of_the_product (id_of_the_product)"

Only use exact names and ids of the products returned as results when providing your final answer.


User prompt:
{input}

{agent_scratchpad}

'''

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
        
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        #tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        kwargs["entity_types"] = json.dumps(entity_types)
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=prompt_template,
    tools=tools,
    input_variables=["input", "intermediate_steps"],
)



from typing import List, Union
import re

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()




from langchain.llms.ollama import Ollama
from langchain import LLMChain
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


groq_llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=groq_llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\Observation:"], 
    allowed_tools=tool_names
)



agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def agent_interaction(user_prompt):
    agent_executor.run(user_prompt)
    
prompt1 = "I'm searching for pink shirts"
agent_interaction(prompt1)

prompt2 = "Can you help me find a toys for my niece, she's 8"
agent_interaction(prompt2)

prompt3 = "I'm looking for nice curtains"
agent_interaction(prompt3)