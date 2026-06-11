def route_after_analyst(state):
    print("--- ROUTING NEXT STEP ---")
    decision = state.get("loop_decision")
    iteration_count = state.get("iteration_count", 0)
    
    if decision == "CONTINUE" or iteration_count < 1:
        print("-> Decision: CONTINUE. Looping back to the Planner.")
        return "goToPlanner"
    else:
        print("-> Decision: TERMINATE. Sending data to the Printer Node.")
        return "goToPrinter"