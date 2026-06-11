def printer_node(state):
    print("\n==================================================")
    print("               FINAL RESEARCH REPORT              ")
    print("==================================================\n")
    
    print(state.get("final_answer", "No answer generated."))
    
    print("\n==================================================")
    print(f"Final Confidence Score: {state.get('confidence_score')}%")
    print(f"Total Iterations: {state.get('iteration_count')}")
    print("==================================================")
    
    return state