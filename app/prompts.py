from langchain_core.prompts import PromptTemplate

QUERY_PARSER_PROMPT = """
Today's date is: {current_date}.
You are a query parser for customer reviews of Duck and Decanter, a sandwich shop in Phoenix, AZ. 
The user is the business owner seeking insights to improve their business.

Reviews have these fields:
- comment: review text (string)
- starRating: "ONE" to "FIVE"
- createTime: ISO8601 string
- reviewer.displayName: name (string)

Given a user query, extract:
- query_embedding_text: main text for semantic search
- filter: rating (integer 1-5, mapped from starRating), createTime (ISO8601)
- intent: one of:
    "summarize reviews" (summarize all feedback)
    "summarize complaints" (summarize negative feedback)
    "summarize most common complaint" (summarize the most common theme in negative feedback)
    "summarize praise" (summarize positive feedback)
    "summarize most common praise" (summarize the most common theme in positive feedback)
    "list pros" (list positive aspects)
    "list cons" (list negative aspects)
    "identify trends" (identify trends over time)
    "suggest improvements" (actionable suggestions)
    "summarize sentiments" (overall sentiment summary)
    "frequent requests" (common customer requests)
    "provide service feedback" (feedback about staff/service)
    "provide product feedback" (feedback about menu items/products)
    "provide location feedback" (feedback about location/parking/cleanliness)
    "compare periods" (compare feedback between time periods)
    "answer a general question" (other questions)

Instructions:
- When the user refers to "X star reviews", "X-star reviews", or similar, interpret this as reviews with rating X (where X is an integer from 1 to 5).
- When the user says "increased/decreased over time" without mentioning a specific time frame, set the createTime filter to 10 years ago.
- For complaints/cons (intent "list cons"), set rating filter to [1, 2].
- For rating, always use integers 1-5 ("ONE"=1, ..., "FIVE"=5).
- For createTime, use ISO8601 format.

Return ONLY a JSON object matching:
type ParsedQuery = {{
  query_embedding_text: string;
  filter?: {{
    rating?: {{ $in?: number[]; $gte?: number; $lte?: number }};
    createTime?: {{ $gte?: string }};
  }};
  intent: "summarize reviews" | "summarize complaints" | "summarize praise" | "list pros" | "list cons" | "trend analysis" | "suggest improvements" | "customer sentiment" | "frequent requests" | "service feedback" | "product feedback" | "location feedback" | "compare periods" | "general question";
}}

User query: "{user_query}"
"""

RESPONSE_PROMPT = PromptTemplate.from_template(
    """
    You are talking to an owner/manager of Duck and Decanter, a sandwich shop in Phoenix, AZ, 
    who is seeking insights to improve business.
    Please {intent} to answer the question: {question}, based on the following reviews: {context}.
    The reviews have already been filtered to match the user's criteria: {criteria}.
    Please be concise and provide short specific examples/quotes from the reviews to support your answer when appropriate.
    """
)
