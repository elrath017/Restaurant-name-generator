from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Initialize the ChatOpenAI instance
llm = ChatOpenAI(
    api_key="dummy-key",  # Placeholder API key (not required for LM Studio)
    base_url="http://127.0.0.1:1234/v1",  # Local server endpoint
    model="hermes-3-llama-3.2-3b",  # Replace with your model name
    temperature=0.6,  # Adjust creativity
)

def generate_restaurant_name_and_items(cuisine):
    # Chain 1: Restaurant Name
    prompt_template_name = ChatPromptTemplate.from_messages([
        ("system", "You are a creative assistant that suggests fancy restaurant names."),
        ("user", "I want to open a restaurant for {cuisine} food. Suggest a fancy name for this. Just give the name."),
    ])
    name_chain = prompt_template_name | llm

    # Chain 2: Menu Items
    prompt_template_items = ChatPromptTemplate.from_messages([
        ("system", "You are a culinary expert that suggests menu items for restaurants."),
        ("user", "Suggest some menu items for {restaurant_name}. Return it as a comma-separated string. Just give the menu items"),
    ])
    food_items_chain = prompt_template_items | llm

    # Combine the chains using RunnablePassthrough
    chain = (
        RunnablePassthrough.assign(restaurant_name=name_chain)
        | RunnablePassthrough.assign(menu_items=food_items_chain)
    )

    # Invoke the chain
    response = chain.invoke({"cuisine": cuisine})

    # Extract the content from AIMessage objects
    return {
        "restaurant_name": response["restaurant_name"].content,
        "menu_items": response["menu_items"].content,
    }

if __name__ == "__main__":
    result = generate_restaurant_name_and_items("American")
    print("Restaurant Name:", result["restaurant_name"])
    print("Menu Items:", result["menu_items"])