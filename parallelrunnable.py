from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

# Components
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Two different prompts
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

# Input
topic = "Machine Learning"


# dictionary is not raunnable so we can use  RunnableParrallel 
# for single topic =>
# chain = RunnableParallel({
#     "short" :short_prompt | model | parser ,
#     "detailed" :detailed_prompt |model |parser
# })

# here no use of RunnableLambda because we are not extracting any value from the input dictionary and passing it to the respective prompt because we are directly passing the topic to the prompt 

# result = chain.invoke({"topic":"Machine Learning"})
 

# for different topic in short and long => we have to use RunnableLambda to extract the topic for short and detailed prompt from the input dictionary of chain and then pass it to the respective prompt and then connect it to the model and parser
chain = RunnableParallel({
    "short" :RunnableLambda(lambda x :x['short']) |short_prompt | model | parser ,
    "detailed" :RunnableLambda(lambda x: x['detailed']) |detailed_prompt |model |parser
})
# for different topic in short and long =>
result = chain.invoke({
    "short" : {"topic":"Machine Learning"},  # here we creating dictionary for topic for short and detailed prompt because we can send more than one topics for short/long  like topic1, topic2 etc and we can create dictionary for each topic and send it to the chain , here only one topic -> topic for short and topic for detailed but we can send more than one topic for short and long and create dictionary for each topic and send it to the chain
    "detailed" : {"topic":"Deep Learning"}
})

print(result['short'])
print(result['detailed'])