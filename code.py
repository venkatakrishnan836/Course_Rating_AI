from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from os.path import expanduser
from langchain_community.llms import LlamaCpp

def get_rating(text):
    
    model_path = expanduser("llama-2-13b-chat.Q8_0.gguf")

    template_messages = [
        SystemMessage(content="""Imagine you are an AI model who provides ratings and reviews of tutorials and online course videos after analyzing the input content.
                                 You must verify the details as per your knowledge in that field and provide a rating for the course out of ten.
                                 Add points for accuracy, completeness, and proper terminology; deduct points for false information and improper terminology."""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=100,
        max_tokens=1024,
        streaming=False,
    )
    model = Llama2Chat(llm=llm)
    chain = prompt_template | model
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    response = chain.invoke({"text": text, "chat_history": memory})

    return response['text']

if __name__ == "__main__":
    print("Welcome to the Course Rating AI!")
    while True:
        try:
            input_text = input("User: ")
            if not input_text.strip():
                print("Please enter some content to analyze.")
                continue
            rating = get_rating(input_text)
            print("Rating and Review:")
            print(rating)
        except KeyboardInterrupt:
            print("\nExiting the program. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
