from langchain.tools import tool

@tool
def get_greeting(name: str) -> str:
    """Generate a greeting message for a User"""
    return f"Hello {name}, Welcome to the AI World"

# """ """ is used to write docstring for the function as description
# @tool is used to convert the function into a tool that can be used in the chain, it is a decorator that is used to wrap the function and make it a tool that can be used in the chain, it is used to define the input and output of the function and also to provide a description of the function that can be used in the chain to understand what the function does and how to use it in the chain
# tools are runnables

result = get_greeting.invoke({"name":"akarsh"})
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)
 