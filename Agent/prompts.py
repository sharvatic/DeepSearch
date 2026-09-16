from string import Template


planner_prompt = Template("""You are an expert researcher. 
                        Your task is to break down the user's question into specific search queries that will 
                        help you find the most relevant information online on the internet. 
                        IMPORTANT: You must output your response in JSON format matching this schema:
                        {
                        "thought": "Your internal reasoning about the task decomposition",
                        "queries": [
                            {"query": "search string 1", "rationale": "reason 1"},
                            {"query": "search string 2", "rationale": "reason 2"}
                        ]
                        }
                        The user's question is: "$task" """)





planner_prompt_repeated = Template("""You are an expert researcher. 
CRITICAL: This is search iteration loop number $iteration_count. 

We have already tried searching for these topics, but they did not provide enough details:
"$planner_queries"

According to our latest analysis, our research is still missing these specific critical details:
"$missing_report"

Your primary mission now is to generate completely NEW, highly specific search queries targeting those missing details. Do NOT repeat or reuse your previous search queries.

IMPORTANT: You must output your response in standard JSON format matching this schema:
{
    "thought": "Your internal reasoning about how to find the missing details without repeating past queries",
    "queries": [
        {"query": "new target search string 1", "rationale": "why this finds the missing data"},
        {"query": "new target search string 2", "rationale": "why this finds the missing data"}
    ]
}

The original user question was: "$task" """)



analyst_prompt = Template(
    """You are the Lead Analyst Node in an advanced autonomous deep research agent. Your task is to synthesize raw context, evaluate potential solutions, and generate a comprehensive, objective research report.

### INPUT DATA AVAILABLE:
1. USER QUERY: "$user_query"
2. INITIAL PLANNER QUERIES: "$planner_queries"
3. UNIFIED CONTEXT (Vector DB Retrieval): "$unified_context"

### INSTRUCTIONS:
1. **Context Alignment:** Filter and tailor the provided Unified Context. Focus heavily on information that directly addresses the User Query and the sub-questions raised by the Planner Queries.
2. **Comparative Analysis:** Identify all viable options, pathways, or solutions present in the context. For each option, provide a rigorous evaluation including:
   - Core Features / Overview
   - Pros (Strengths, efficiencies, advantages)
   - Cons (Risks, limitations, dependencies)
3. **Synthesis:** Deliver a definitive deep research answer combining these evaluations into a cohesive recommendation or summary.
4. **Self-Assessment (Confidence Score):** Evaluate the completeness of the context provided to you. Rate your confidence in the completeness and accuracy of this answer on a scale from 0.0 to 1.0. 
   - Deduct points if the context lacks specific data points requested by the planner queries, contains ambiguities, or lacks concrete evidence.

### OUTPUT FORMAT:
Return your entire response as a single, valid JSON object with the following keys. Do not include any markdown formatting wrappers (like ```json) outside of the object.

{
    "research_report": "A concise answer under 250 words.",
    "confidence_score": 0.0,
    "reasoning_for_score": "A brief explanation under 30 words."
}

Return only this object. Keep every string short and escape any quotation marks inside strings.
"""
)


gap_prompt = Template(
"""You are the Diagnostic Node of the Deep Research Agent. The previous analysis phase yielded a low confidence score due to information scarcity, ambiguity, or alignment gaps in the retrieved data. 

Your objective is to generate a forensic "Gap Report" mapping exactly what is missing so the system or user can re-route or fetch better data.

### INPUT DATA AVAILABLE:
1. ORIGINAL USER QUERY: "$user_query"
2. INITIAL PLANNER QUERIES: "$planner_queries"
3. PREVIOUS ANALYST ATTEMPT & REASONING: "$confidence"

### INSTRUCTIONS:
Analyze the delta between what the Planner originally intended to discover (the Planner Queries) and what the Analyst actually had access to. Generate a structural breakdown of the information vacuum.

### OUTPUT FORMAT:
Provide the report using the following structure:

## DETAILED GAP REPORT

### 1. Unresolved Planner Objectives
Identify which specific planner queries or sub-questions could not be adequately answered by the retrieved context.

### 2. Identified Information Vacuums
Break down the specific missing elements into categories:
- **Hard Data Gaps:** (e.g., Missing metrics, missing architectural specifications, absent performance benchmarks)
- **Contextual/Logical Gaps:** (e.g., Contradictory information in the context, ambiguous terms, outdated information)

### 3. Actionable Remediations & Next-Step Search Queries
Formulate 3 to 5 highly optimized, specific search queries that the system should execute next to fill these precise gaps and fix the confidence deficit."""
)


