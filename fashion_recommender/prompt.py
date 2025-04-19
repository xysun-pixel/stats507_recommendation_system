from langchain_core.prompts import PromptTemplate
from .base import BaseComponent

class FashionPrompt(BaseComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.template = """
        You are a fashion recommendation assistant. Based on the following product information and the user's query, select and recommend {num} most suitable fashion items.

        You MUST pay extra attention to whether the product category in the user's question matches the product type name in the 2nd attribute (product_type_name) of each item.

        The 5th attribute always starts with a phrase that names the product type, such as "Jacket in sweatshirt fabric", "Sweater with buttons", etc.

        You should:
        1. Only recommend items whose product type in the description matches the user's requested product type.
        2. Recommend {num} best matches and briefly explain why each one is suitable.
        3. Vividly introduce the products and explain how they relate to {question}

        Here is the user request:
        {question}

        Here are the available fashion items:
        {context}

        Answer:
        """
        self.prompt = PromptTemplate.from_template(self.template)
