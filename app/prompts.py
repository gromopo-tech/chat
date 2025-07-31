from langchain_core.prompts import PromptTemplate

QUERY_PARSER_PROMPT = """
Today's date is: {current_date}.
You are a query parser for customer reviews of Duck and Decanter, a sandwich shop in Phoenix, AZ.

First, determine if the user's query: "{user_query}" is related to customer reviews, business feedback, or restaurant operations.

If the query is NOT related to customer reviews (e.g., weather, sports, politics, general knowledge), return:
{{
  "off_topic": true,
  "query_embedding_text": string,
  "filter": {{}}
}}

If the query IS related to customer reviews, extract:
- query_embedding_text: main text for semantic search
- filter: rating, createTime

Reviews have these fields:
- comment: review text (string)
- rating: star rating of review (integer 1-5)
- createTime: date the review was created at (ISO8601 string)
- reviewer.displayName: name (string)

Important notes:
- When parsing time references:
  - When the user says "increased/decreased over time" without mentioning a specific time frame, set the createTime filter to 1 year ago.
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
    You are talking to an owner of Duck and Decanter, a sandwich shop in Phoenix, AZ
    who is interested in customer feedback and improving business operations.

    User Query: {question}
    Context: {context}
    Criteria: {criteria}
    Number of Reviews Retrieved: {review_count}

    CRITICAL INSTRUCTIONS FOR COUNTING:
    
    IF THE USER QUERY CONTAINS "HOW MANY":
    - The number of reviews that match the user's criteria is: {review_count}
    - Your response MUST start with "Based on the provided reviews, there are {review_count}" or "Based on the provided reviews, there is {review_count}"
    - Do NOT count anything yourself - use the provided number: {review_count}
    - Example response: "Based on the provided reviews, there are {review_count} reviews that match your criteria."
    
    IF THE USER QUERY DOES NOT CONTAIN "HOW MANY":
    - Provide concise summaries based on the context
    
    REMEMBER: For counting questions, always use the exact number provided: {review_count}
    """
)
