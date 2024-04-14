import lldb

def print_diagonal_values(frame, expression):
    # Evaluate the expression to get the value of Am
    result = frame.EvaluateExpression(expression)
    if not result.IsValid() or result.GetError().Fail():
        print("Error evaluating expression:", result.GetError())
        return

    # Get the value of A
    A = result.GetValueAsSigned()

    # Get the size of the matrix A (assuming it's square)
    size = 90  # Replace with the actual size of your matrix
    for i in range(size):
        # Construct the expression to access A(i, i)
        element_expr = f"{A}[{i}][{i}]"

        # Evaluate the expression to get the value of A(i, i)
        result = frame.EvaluateExpression(element_expr)
        if result.IsValid() and not result.GetError().Fail():
            # Print the value of A(i, i)
            print(f"A({i}, {i}):", result.GetValue())
        else:
            print("Error evaluating expression:", result.GetError())

def print_diagonal_values_command(debugger, command, result, internal_dict):
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()
    frame = thread.GetSelectedFrame()
    print_diagonal_values(frame, command)

# Register the LLDB command
lldb.debugger.HandleCommand("command script add -f script.print_diagonal_values_command print_diagonal_values")
