from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output Parser
parser = StrOutputParser()

# connecting runnables in asequence -> b/c now no chain runnable is available in langchain, we can use the | operator to connect them in a sequence
# for runnables we use invoke() method to execute them, and for chain we can use the same invoke() method to execute the whole chain
# it is sequence runnable

chain = prompt | model | parser

result = chain.invoke("Machine Learning")
print(result)
