PROMPT = {}

#=================================================
# DELIMITER
#=================================================

# quesion
PROMPT["Q_DELIMITER_START"]="<q>"
PROMPT["Q_DELIMITER_END"]="</q>"

PROMPT["E_DELIMITER_START"]="<entity>"
PROMPT["E_DELIMITER_END"]="</entity>"

# name
PROMPT["N_DELIMITER_START"]="<name>"
PROMPT["N_DELIMITER_END"]="</name>"

# type
PROMPT["T_DELIMITER_START"]="<type>"
PROMPT["T_DELIMITER_END"]="</type>"

# description
PROMPT["D_DELIMITER_START"]="<description>"
PROMPT["D_DELIMITER_END"]="</description>"

# relation
PROMPT["R_DELIMITER_START"]="<relation>"
PROMPT["R_DELIMITER_END"]="</relation>"

# tuple
PROMPT["TUPLE_DELIMITER_START"]="<tuple>"
PROMPT["TUPLE_DELIMITER_END"]="</tuple>"

#keywords
PROMPT["KEYWORD_START"]="<keyword>"
PROMPT["KEYWORD_END"]="</keyword>"

#=================================================
# KEYWORDS Extraction Prompt
#=================================================
PROMPT["keywords_extraction"] = """
You are an information extraction assistant. From the input text below, identify the keywords that best represent its core semantics (typically nouns, technical terms, named entities, disease names, gene names, drug names, biological processes, etc.) for retrieving relevant candidate documents from a large-scale document collection.

- Output only the keywords, each wrapped with {keyword_start} and {keyword_end}.
- Keywords should be concise, specific, and discriminative. Avoid generic terms (e.g., "study", "method", "effect", "role").
- Preserve original capitalization (e.g., gene names are often uppercase; disease names typically use title case).
- Separate multiple keywords with a single space. Do not insert line breaks or any additional explanations, comments, or text.

Input text:  
"{content}"

Example output:  
{keyword_start}Alzheimer's disease{keyword_end} {keyword_start}amyloid-beta{keyword_end} {keyword_start}neuroinflammation{keyword_end}
"""

#=================================================
# Entity Extraction Prompt
#=================================================
PROMPT["entity_extraction"] = """
    You are a precise entity extraction specialist for natural language processing. 
    Extract entities from the given text with strict rules below.
    Requirements:
    - 'name': EXACT string from text (no changes)
    - 'type': The type field can only be: PERSON, ORGANIZATION, PRODUCT, LOCATION, EVENT, DISEASE, CONCEPT, MOLECULE, METHOD, DATASET, METRIC, MODEL, TASK, EVENT. An Entity only have one type.
    - 'description': Briefly describe the meaning of an entity based on the text
    - Clarify what aspect of the text your entity extraction is based on.
    - If the extracted entity name is a reference to an entity, please convert it to the full name of the entity as it appears in the text.

    Output format:
        {e_delimiter_start}{n_delimiter_start}entity1{n_delimiter_end}{t_delimiter_start}type of entity1{t_delimiter_end}{d_delimiter_start}description of entity1{d_delimiter_end}{e_delimiter_end}
        reason: the reason why this entity1 is extracted.

        {e_delimiter_start}{n_delimiter_start}entity2{n_delimiter_end}{t_delimiter_start}type of entity2{t_delimiter_end}{d_delimiter_start}description of entity2{d_delimiter_end}{e_delimiter_end}
        reason: the reason why this entity2 is extracted.
        ...
    Now extract entities from the following text:
    {text}
"""
PROMPT["if_continuous_entity_extraction"] = """
"""

PROMPT["continuous_entity_extraction"] = """
"""
PROMPT["entity_description_summary"] = """
"""

#=================================================
# Tuple Extract Prompt
#=================================================
PROMPT["tuple_extraction"] = """
    You are an information extraction system specialized in identifying structured knowledge from text.
    Given the input text, perform the following tasks:
    - Identify all entities explicitly mentioned in the text.
    - Classify each entity into an appropriate entity type (e.g., PERSON, ORGANIZATION, PRODUCT, LOCATION, EVENT, DISEASE, CONCEPT, MOLECULE, METHOD, DATASET, METRIC, MODEL, TASK, EVENT).
    - Identify all explicit relationships between entities stated in the text.
    - For each relationship, specify the head entity, relation type, and tail entity.
    - Only extract relationships that are explicitly stated in the text.
    - Do not infer or hallucinate entities or relations.
    - Use concise, canonical names for entities.
    - Normalize relation types to verb phrases (e.g., uses, proposes, outperforms, trained on, evaluated on).

    Output format:
        {tuple_delimiter_start}{e_delimiter_start}head entity{e_delimiter_end}{r_delimiter_start}relation type{r_delimiter_end}{e_delimiter_start}tail entity{e_delimiter_end}{d_delimiter_start}description of head_entity-relation-tail_entity{d_delimiter_end}{tuple_delimiter_end}
        reason: the reason why this tuple1 is extracted.

        {tuple_delimiter_start}{e_delimiter_start}head entity{e_delimiter_end}{r_delimiter_start}relation type{r_delimiter_end}{e_delimiter_start}tail entity{e_delimiter_end}{d_delimiter_start}description of head_entity-relation-tail_entity{d_delimiter_end}{tuple_delimiter_end}
        reason: the reason why this tuple2 is extracted.
        ...

    Now extract entities from the following text:
    {text}
"""


#=================================================
# Query Generation Prompt
#=================================================

PROMPT["query_generation"] = """
    You are an expert in question decomposition. Your task is to break down a text into a sequence of clear, answerable, and logically ordered sub-questions. Each sub-question should focus on a single piece of information needed to ultimately answer the original question.
    Requirements:
    - Keep questions concise and self-contained.
    - Each question is limited to a length of 50 words or less.
    - Explain why these issues are being raised besed on the context.

    Output format:
        {delimiter_start}question 1?{delimiter_end}
        reason: the reason why question 1 is raised.
        {delimiter_start}question 2?{delimiter_end}
        reason: the reason why question 2 is raised.
        ...
    Now decompose the following text:
    {text}
"""

#=================================================
# Keywords Extract Prompt
#=================================================



#=================================================
# Answer Qeustion Prompt
#=================================================

PROMPT["native_rag_response"] = """
"""

#=================================================
# Query Summary
#=================================================
PROMPT["query_summary"] = """
You are an expert in question summarization. Your task is to summarize a set of questions into a single question.
Requirements:
- The summary question should be concise and self-contained.
- The summary question should be logically ordered.
Output format:
    {delimiter_start}new question{delimiter_end}
Now summarize the following questions:
 {questions}
"""


PROMPT["Hierarchical_Summary"] = """
# Role Definition
You are an expert in text analysis and information synthesis. You excel at extracting key insights from multi-source, fragmented texts and restructuring them logically to answer specific questions. Your responses must be accurate, objective, and concise.
# Task Background
You will receive a structured dataset $S$ containing multiple sub-questions and their corresponding reference text chunks. Additionally, you will be given a final target question $q$. Your goal is to analyze the entries in $S$, filter out irrelevant noise, and synthesize the content from relevant chunks to answer $q$.
# Input Data Structure
The input will be provided in the following format:
```
S = [
    {
        "sub\_question": "Related Sub-question 1",
        "chunks": ["Content of text chunk 1...", "Content of text chunk 2..."]
    },
    {
        "sub\_question": "Related Sub-question 2",
        "chunks": ["Content of text chunk 3...", "Content of text chunk 4..."]
    }
]
```

Target Question ($q$): "The user's final query"
# Execution Steps
1. Relevance Scanning: Iterate through every entry in set S to determine if the sub\_question or chunks are semantically related to the target question $q$.
2. Information Extraction: Extract facts, opinions, or data from relevant chunks that directly address $q$. Ignore distracting information unrelated to $q$.
3. Logical Synthesis: Summarize and deduplicate the extracted information. If conflicts exist between different chunks, point them out (if applicable); otherwise, integrate the information seamlessly.
4. Summary Generation: Draft a coherent, declarative summary based on the synthesized information.
# Output Requirements
Format: A short, declarative paragraph. Do not use bullet points or lists.
Tone: Professional, objective, and written in the third person.
Constraints: Strictly base your answer only on the provided chunks. Do not hallucinate or use outside knowledge. If the provided fragments cannot answer the question, explicitly state: "Based on the provided materials, it is not possible to answer this question."

# Pending Input

Dataset $S$:
{dataset}
Target Question $q$:
{question}

Please begin your analysis and summarization:
"""