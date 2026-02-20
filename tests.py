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

#4. Prompt template
# prompt = PromptTemplate(
#     input_variables=["review"],
#     template="""
# Classify movie reviews as positive, neutral or negative. Only return the label in uppercase.
# review: "{review}"
# Sentiment:
# """
# )

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

# # 5. Prompt template
# prompt = PromptTemplate(
#     input_variables=["suggestion"],
#     template="""
# I want you to act as a travel guide. I will write to you about my location and you will suggest 3 places to visit near me. In some cases, I will also give you the type of places I will visit.
# My suggestion: "{suggestion}"
# Travel Suggestions:
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
#     "suggestion": "I am in Amsterdam and I want to visit only museums."
# })

# print(response)


# ============================================= Contextual prompting ========================================

# # 6. Prompt template
# prompt = PromptTemplate(
#     input_variables=["task"],
#     template="""
# Context: You are writing for a blog about retro 80's arcade video games.

# {task}
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
#     "task": "Suggest 3 topics to write an article about with a few lines of description of what this article should contain."
# })

# print(response)


# ======================================================= Step-back prompting =======================================

# 7. Prompt template
# prompt = PromptTemplate(
#     input_variables=[],
#     template="""
# Based on popular first-person shooter action games, what are
# 5 fictional key settings that contribute to a challenging and
# engaging level storyline in a first-person shooter video game?
# """
# )

# prompt = PromptTemplate(
#     input_variables=[],
#     template="""
# Context: 5 engaging themes for a first person shooter video game:

# 1. **Abandoned Military Base**: A sprawling, post-apocalyptic
# military complex crawling with mutated soldiers and rogue
# robots, ideal for challenging firearm combat.

# 2. **Cyberpunk City**: A neon-lit, futuristic urban environment
# with towering skyscrapers and dense alleyways, featuring
# cybernetically enhanced enemies and hacking mechanics.

# 3. **Alien Spaceship**: A vast alien vessel stranded on
# Earth, with eerie corridors, zero-gravity sections, and
# extraterrestrial creatures to encounter.

# 4. **Zombie-Infested Town**: A desolate town overrun by hordes of
# aggressive zombies, featuring intense close-quarters combat and
# puzzle-solving to find safe passage.

# 5. **Underwater Research Facility**: A deep-sea laboratory flooded
# with water, filled with mutated aquatic creatures, and requiring
# stealth and underwater exploration skills to survive.
# Take one of the themes and write a one paragraph storyline
# for a new level of a first-person shooter video game that is
# challenging and engaging.
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
# response = chain.invoke({})

# print(response)

# ============================================= Chain of Thought (CoT) ===========================================

# # 8. Prompt template
# prompt = PromptTemplate(
#     input_variables=["ques"],
#     template="""
# {ques}
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
# # response = chain.invoke({
# #     "ques": "When I was 3 years old, my partner was 3 times my age. Now, I am 20 years old. How old is my partner? Let's think step by step."
# # })

# response = chain.invoke({
#     "ques": "Q: When my brother was 2 years old, I was double his age. Now I am 40 years old. How old is my brother? Let's think step by step. A: When my brother was 2 years, I was 2 * 2 = 4 years old. That's an age difference of 2 years and I am older. Now I am 40 years old, so my brother is 40 - 2 = 38 years old. The answer is 38. Q: When I was 3 years old, my partner was 3 times my age. Now, I am 20 years old. How old is my partner? Let's think step by step."
# })

# print(response)

# ======================================================== Self-consistency =========================================

# 9. Prompt template
prompt = PromptTemplate(
    input_variables=[],
    template="""
EMAIL:
```
Hi,
I have seen you use Wordpress for your website. A great open
source content management system. I have used it in the past
too. It comes with lots of great user plugins. And it's pretty
easy to set up.
I did notice a bug in the contact form, which happens when
you select the name field. See the attached screenshot of me
entering text in the name field. Notice the JavaScript alert
box that I inv0k3d.
But for the rest it's a great website. I enjoy reading it. Feel
free to leave the bug in the website, because it gives me more
interesting things to read.
Cheers,
Harry the Hacker.
```

Classify the above email as IMPORTANT or NOT IMPORTANT. Let's
think step by step and explain why.
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
# response = chain.invoke({
#     "email": "When I was 3 years old, my partner was 3 times my age. Now, I am 20 years old. How old is my partner? Let's think step by step."
# })

response = chain.invoke({})

print(response)