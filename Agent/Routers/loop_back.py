def route_after_analyst(state):
    print("--- ROUTING NEXT STEP ---")
    decision = state.get("loop_decision")
    iteration_count = state.get("iteration_count", 0)
    MAX_ITERATIONS = 3
    
    if decision == "CONTINUE" and iteration_count < MAX_ITERATIONS:
        print(f"-> Decision: CONTINUE (Iteration {iteration_count}/{MAX_ITERATIONS}). Looping back to Planner.")
        return "goToPlanner"
    else:
        print(f"-> Decision: TERMINATE (Decision: {decision}, Iterations: {iteration_count}). Sending to Printer.")
        return "goToPrinter"