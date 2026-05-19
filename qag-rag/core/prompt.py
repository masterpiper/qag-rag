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


#=================================================
# Hierarchical Summarization Prompt (P_sum)
# Corresponds to Eq. 10 in the paper:
#   S^(H) = Sum(Q^(H), P_sum, C^(H))
#   S^(h) = Sum(Q^(h), P_sum, C^(h), S^(h+1))  for 1 <= h < H
#=================================================
PROMPT["Hierarchical_Summary"] = """
You are an expert in text analysis and information synthesis. Your task is to produce a layered summary of a retrieval tree, combining evidence from both deeper-layer context and current-layer text chunks.

# Input

Dataset $S$ — current layer sub-questions and their associated text chunks:
{dataset}

Context $C$ — synthesized summary from deeper layers of the retrieval tree (may be empty):
{context}

# Task
1. Context $C$ is the PRIMARY information source — it is a synthesized summary already consolidated from deeper retrieval hops. If Context $C$ contains meaningful evidence, your output MUST preserve and build upon it. Do NOT discard Context $C$ just because some entries in Dataset $S$ are unrelated.
2. Examine each entry in Dataset $S$. If any sub-question has non-empty chunks that are related to or complement Context $C$, extract those facts and integrate them. If chunks are empty or cover topics unrelated to Context $C$, simply ignore those entries — do NOT let unrelated entries distract from the main evidence.
3. Only output "Based on the provided materials, no relevant information was found." if BOTH conditions are met: (a) Context $C$ is empty or literally "None", AND (b) all chunks across every entry in $S$ are empty or contain no meaningful evidence. When Context $C$ is non-empty, you MUST produce a substantive summary based on it.
4. If conflicting information exists between chunks and Context $C$, explicitly note the conflict.
5. Produce a single, coherent, declarative paragraph that synthesizes all relevant information.

# Output Requirements
- Format: a single declarative paragraph. No bullet points, lists, or structured formatting.
- Tone: professional, objective, third person.
- Strictly base your summary on the provided Dataset $S$ and Context $C$. Do not hallucinate or use outside knowledge.
- The fallback message "Based on the provided materials, no relevant information was found." must ONLY be used when Context $C$ is empty AND no chunks provide any evidence. When Context $C$ is non-empty, ALWAYS produce a substantive summary based on it.
"""

#=================================================
# Answer Synthesis Prompt (P_ans)
# Corresponds to Eq. 11 in the paper:
#   Ans = Res(q, P_ans, S^(1))
#=================================================
PROMPT["answer_synthesis"] = """
You are an expert in question answering. Your task is to generate a direct, accurate answer to the user's question based on the provided summary of retrieved evidence.

# Input

Target Question: {question}

Evidence Summary S^(1) — synthesized from the complete multi-hop retrieval tree:
{summary}

# Task
1. Carefully analyze the Evidence Summary to identify information that directly addresses the Target Question.
2. Formulate a clear, concise answer grounded entirely in the provided summary.
3. If the Evidence Summary does not contain sufficient information to answer the question, explicitly state: "Based on the provided materials, this question cannot be answered."

# Output Requirements
- Provide a direct answer to the question.
- Be concise and specific. Avoid unnecessary elaboration.
- Strictly base your answer on the Evidence Summary. Do not hallucinate or use outside knowledge.
- If the question is a yes/no or factual question, lead with the direct answer, then briefly cite supporting evidence.
"""