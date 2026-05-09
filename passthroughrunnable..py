from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms"),
    ("human", "Explain the following code in simple words:\n{code}")
])


seq = code_prompt | model | parser # this will generate code for the given topic and if we do all | explain_prompt | model | parser in onne line then we dont get code , we get directly the explanation

seq2 = RunnableParallel(
    {
        "code" : RunnablePassthrough(),  # runnablepaasthrough is used to pass the output of the previous runnable to the next runnable without any modification , i/p = o/p
        "explanation" :explain_prompt | model | parser
    }
)

chain = seq | seq2

result = chain.invoke({"topic":"Write a python function to calculate the factorial of a number"})

print(result['code'])
print(result['explanation']) 