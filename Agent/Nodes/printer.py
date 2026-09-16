import pprint


def printer_node(state):
    print("\n==================================================")
    print("               FINAL RESEARCH REPORT              ")
    print("==================================================\n")
    
    pprint.pprint(state.get("final_answer", "No answer generated."), indent=2, width=80)
    
    score = state.get("confidence_score", 0.0)
    score_pct = score * 100 if score <= 1.0 else score
    
    print("\n==================================================")
    print(f"Final Confidence Score: {score_pct:.1f}%")
    print(f"Total Iterations: {state.get('iteration_count')}")
    print("==================================================")
    
    return state