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



analyst_prompt = Template("""You are an elite research analyst evaluating information for the main query: "$user_query"

                            Below is the verified context extracted from our web search database:
                            "$unified_context"

                            Your task is to analyze this context, compile a factual summary, discover what critical data points are STILL missing, and assign a confidence score to your current understanding.

                            IMPORTANT: You must output your analysis in standard JSON format matching this schema:
                            {
                                "current_summary": "A detailed synthesis of the facts currently discovered.",
                                "missing_details_report": "A precise list of critical metrics, figures, or facts that are still missing from the context.",
                                "confidence_score": the confidence score (0-100) representing how complete and reliable the current understanding is based on the context,
                                "loop_decision": "CONTINUE" or "TERMINATE"
                            }

                            CRITICAL CONDITION FOR loop_decision:
                            - Set to "TERMINATE" ONLY if confidence_score >= 80 OR iteration_count > 2.
                            - Otherwise, set to "CONTINUE". Current iteration count is: "$iteration_count" """)



