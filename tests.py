import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ============================================================ Zero shot prompting =========================================

# 1. Prompt template
# prompt = PromptTemplate(
#     input_variables=["Review"],
#     template="""
# Classify movie reviews as POSITIVE, NEUTRAL or NEGATIVE.
# Review: "{Review}"
# Sentiment:
# """
# )

# # 2. Gemini model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=1,max_output_tokens=1024,top_k=40,top_p=0.8)

# # 3. Output parser
# output_parser = StrOutputParser()

# # 4. Build chain
# chain = RunnableSequence(
#     prompt | model | output_parser
# )

# # 5. Invoke chain
# response = chain.invoke({
#     "Review": "Her is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. I wish there were more movies like this masterpiece."
# })

# print(response)

# ============================================================ One shot prompting =================================================================

# 2. Prompt template
# prompt = PromptTemplate(
#     input_variables=["order"],
#     template="""
# Parse a customer's pizza order into valid JSON:

# EXAMPLE:
# I want a small pizza with cheese, tomato sauce, and pepperoni.
# JSON Response:
# ```
# {{
# "size": "small",
# "type": "normal",
# "ingredients": [["cheese", "tomato sauce", "peperoni"]]
# }}
# Now parse this order: "{order}"
# JSON Response:
# """
# )

# # 2. Gemini model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=1,max_output_tokens=1024,top_k=40,top_p=0.8)

# # 3. Output parser
# output_parser = StrOutputParser()

# # 4. Build chain
# chain = RunnableSequence(
#     prompt | model | output_parser
# )

# # 5. Invoke chain
# response = chain.invoke({
#     "order": "Now, I would like a large pizza, with the first half cheese and mozzarella. And the other tomato sauce, ham and pineapple."
# })

# print(response)

# ============================================================ Few Shot prompting ===================================================================

# 3. Prompt template
# prompt = PromptTemplate(
#     input_variables=["order"],
#     template="""
# Parse a customer's pizza order into valid JSON.

# EXAMPLE:
# I want a small pizza with cheese, tomato sauce, and pepperoni.
# JSON Response:
# {{
#   "size": "small",
#   "type": "normal",
#   "ingredients": [["cheese", "tomato sauce", "pepperoni"]]
# }}

# EXAMPLE:
# Can I get a large pizza with tomato sauce, basil and mozzarella
# {{
# "size": "large",
# "type": "normal",
# "ingredients": [["tomato sauce", "bazel", "mozzarella"]]
# }}

# Now parse this order:
# {order}

# JSON Response:
# """
# )

# # 2. Gemini model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=0.1,max_output_tokens=250,top_p=1)

# # 3. Output parser
# output_parser = StrOutputParser()

# # 4. Build chain
# chain = RunnableSequence(
#     prompt | model | output_parser
# )

# # 5. Invoke chain
# response = chain.invoke({
#     "order": "Now, I would like a large pizza, with the first half cheese and mozzarella. And the other tomato sauce, ham and pineapple."
# })

# print(response)

# ==================================================== System Prompting ====================================================================

# 4. Prompt template
# prompt = PromptTemplate(
#     input_variables=["review"],
#     template="""
# Classify movie reviews as positive, neutral or negative. Only return the label in uppercase.
# review: "{review}"
# Sentiment:
# """
# )

# # 2. Gemini model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=1,max_output_tokens=1024,top_k=40,top_p=0.8)

# # 3. Output parser
# output_parser = StrOutputParser()

# # 4. Build chain
# chain = RunnableSequence(
#     prompt | model | output_parser
# )

# # 5. Invoke chain
# response = chain.invoke({
#     "review": "Her is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. It's so disturbing I couldn't watch it."
# })

# print(response)

# ==================================================== another example of system prompting ====================================

# # 4. Prompt template
# prompt = PromptTemplate(
#     input_variables=["review"],
#     template="""
# Classify movie reviews as positive, neutral or negative. Return valid JSON:
# Schema:
# ```
# MOVIE:
# {{
# "sentiment": String "POSITIVE" | "NEGATIVE" | "NEUTRAL",
# "name": String
# }}
# MOVIE REVIEWS:
# {{
# "movie_reviews": "{review}"
# }}
# ```
# JSON Response:
# """
# )

# # 2. Gemini model
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=1,max_output_tokens=1024,top_k=40,top_p=0.8)

# # 3. Output parser
# output_parser = StrOutputParser()

# # 4. Build chain
# chain = RunnableSequence(
#     prompt | model | output_parser
# )

# # 5. Invoke chain
# response = chain.invoke({
#     "review": "Her is a disturbing study revealing the direction humanity is headed if AI is allowed to keep evolving, unchecked. It's so disturbing I couldn't watch it."
# })

# print(response)

# ========================================================= Role prompting ==================================================================

# 5. Prompt template
prompt = PromptTemplate(
    input_variables=["suggestion"],
    template="""
I want you to act as a travel guide. I will write to you about my location and you will suggest 3 places to visit near me. In some cases, I will also give you the type of places I will visit.
My suggestion: "{suggestion}"
Travel Suggestions:
"""
)

# 2. Gemini model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",temperature=1,max_output_tokens=1024,top_k=40,top_p=0.8)

# 3. Output parser
output_parser = StrOutputParser()

# 4. Build chain
chain = RunnableSequence(
    prompt | model | output_parser
)

# 5. Invoke chain
response = chain.invoke({
    "suggestion": "I am in Amsterdam and I want to visit only museums."
})

print(response)