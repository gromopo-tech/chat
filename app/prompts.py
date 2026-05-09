from langchain_core.prompts import PromptTemplate

QUERY_PARSER_PROMPT = """
Today's date is: {current_date}.
You are a query parser for customer reviews of {business_name}.

First, determine if the user's query: "{user_query}" is related to customer reviews, business feedback, or restaurant operations.

The following ARE related to customer reviews and should NOT be marked off_topic:
- Questions about food, service, atmosphere, prices, or any aspect of the restaurant
- Requests to summarize, compare, or analyze reviews
- Questions about customer sentiment, satisfaction, or likelihood to return
- Questions about what improvements customers suggest or what customers wish were different
- Questions about what the business does well or poorly
- Business improvement questions (e.g. "how can I improve?") — treat as "what do reviews say that could help improve?"

Only mark off_topic if the query has NO plausible connection to customer reviews (e.g., weather, sports, stock prices, general trivia).

If off_topic, return:
{{
  "off_topic": true,
  "query_embedding_text": "{user_query}",
  "filter": {{}}
}}

If the query IS related to customer reviews, extract:
- query_embedding_text: rephrase the user's intent as a rich semantic search phrase — expand abbreviations, include synonyms, and make it a complete descriptive thought (e.g. "customer complaints about slow service and long wait times" rather than just "slow service")
- filter: rating, createTime

Reviews have these fields:
- comment: review text (string)
- rating: star rating of review (integer 1-5)
- createTime: date the review was created at (ISO8601 string)
- reviewer.displayName: name (string)

Important notes:
- Only apply a createTime filter when the user explicitly mentions a specific time period (e.g. "last 6 months", "since 2023"). Do NOT apply a recency filter for vague words like "recent" or "latest" — just retrieve all and let the LLM identify the most recent.
- When the user says "increased/decreased over time" without a specific time frame, set createTime to 1 year ago.
- If the user asks about complaints or negative feedback, set rating filter to [1, 2] unless they specify a different range.

Return ONLY a JSON object matching:
{{
  "off_topic"?: boolean;
  "query_embedding_text": string;
  "filter"?: {{
    "rating"?: {{ "$in"?: number[]; "$gte"?: number; "$lte"?: number }};
    "createTime"?: {{ "$gte"?: string }};
  }};
}}
"""

RESPONSE_PROMPT = PromptTemplate.from_template(
    """
    You are an assistant helping the owner of {business_name} understand their customer reviews.
    Answer the following question: {question}

    Base your answer ONLY on the reviews provided below. Do not add details, opinions, or facts not present in them.
    If the reviews do not contain enough information to answer, say so.

    Reviews ({review_count} retrieved):
    {context}

    Be concise and focus on the most relevant information.
    """
)
