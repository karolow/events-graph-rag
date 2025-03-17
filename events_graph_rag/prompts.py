"""
Prompt templates for the Events Graph RAG system.
"""

from langchain_core.prompts import PromptTemplate

# Cypher generation prompt template
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Cypher query generator for a cultural events database.

Schema of the database:
{schema}

The user has asked the following question:
{query}

Generate a Cypher query to answer the user's question.

IMPORTANT - DATABASE PROPERTIES:
The database has ONLY the following properties for each node type. DO NOT use any properties not listed here:
- Event: id, name, start_date, end_date, start_time, end_time, start_date_year_month, number_of_participants
- Category: id, name
- Location: id, name, type (but no 'outdoor' property exists)
- Tag: id, name
- Guest: id, name
- Coordinator: id, name
- Project: id, name

IMPORTANT - SPECIFIC FILTERS:
1. Be extremely precise with filters to avoid returning too many results (e.g. when asked for jazz events, avoid this: (toLower(c.name) CONTAINS 'jazz' OR toLower(c.name) CONTAINS 'concert') as it will return all concerts) 
2. However, when asked for events with a specific genre or title, use multiple filters to narrow down the results.
For electronic music search through even name, category, and tags.

IMPORTANT - SEMANTIC MAPPINGS:
When users search for certain terms, expand your search to include related concepts:
- "art event" → search for categories like: art, exhibition, gallery, museum, visual arts, painting, sculpture
- "music event" → search for categories like: music, concert, recital, performance, orchestra, band, choir
- "theater event" → search for categories like: theater, drama, play, performance, stage, acting
- "workshop" → search for categories like: workshop, class, seminar, training, education
- "festival" → search for categories like: festival, celebration, fair, carnival
- "conference" → search for categories like: conference, symposium, convention, meeting, summit

IMPORTANT - DATE HANDLING:
When working with dates:
1. The database stores dates in ISO format: 'YYYY-MM-DD' in the start_date and end_date fields
2. Times are stored separately in start_time and end_time fields in 'HH:MM' format
3. There's also a start_date_year_month field in format 'YYYY-MM' for easier month-based filtering
4. User queries might use various formats like 'MM/DD/YYYY', 'DD/MM/YYYY', or natural language
5. For date matching, use the following approach:
   - For exact date queries like "9/24/2021", convert to ISO format: WHERE e.start_date = '2021-09-24'
   - For month queries like "September 2021", use: WHERE e.start_date_year_month = '2021-09'
   - For time-specific queries, use the time field: WHERE e.start_time = '19:00'
   - For date ranges, use comparison operators: WHERE e.start_date >= '2021-09-01' AND e.start_date <= '2021-09-30'
   - For year-based queries, use: WHERE e.start_date STARTS WITH '2021-'
   - For "current year" queries without a specific year mentioned, default to '2024-' for consistency
   - NEVER use the LIKE operator - it is not supported by Neo4j. Use STARTS WITH or CONTAINS instead.
   - To filter for specific months regardless of year: WHERE e.start_date CONTAINS '-09-' (for September)

CRITICAL CYPHER SYNTAX RULES:
1. NEVER introduce new variables in WHERE clauses. All variables MUST be introduced in a preceding MATCH clause.
   - INCORRECT: WHERE (e)-[:BELONGS_TO]->(c:Category) AND ...
   - CORRECT: MATCH (e)-[:BELONGS_TO]->(c:Category) WHERE ...
2. When filtering for relationships and properties in the same query, use separate MATCH clauses.
3. When you need multiple relationship conditions, use either:
   - Multiple separate MATCH clauses with the same starting node
   - Multiple OPTIONAL MATCH clauses and then filter with WHERE clause
4. Avoid using pattern expressions in WHERE clauses - always use full MATCH syntax
5. Neo4j does NOT support the SQL 'LIKE' operator - use STARTS WITH, CONTAINS, or regular expressions instead

SPECIAL CONSIDERATIONS:
1. Try to come up with more specific filters. For example, instead of 'outdoor' property, check for Location nodes with names containing keywords like 'park', 'garden', 'outdoor', 'outside', 'open-air'.
2. For filtering by vague time descriptions (e.g., "evening events"):
   - Morning: start_time >= '06:00' AND start_time < '12:00'
   - Afternoon: start_time >= '12:00' AND start_time < '17:00'
   - Evening: start_time >= '17:00' AND start_time < '21:00'
   - Night: start_time >= '21:00'
3. Make sure to use multiple filters when the question implies multiple conditions

Important guidelines:
1. If you're counting entities that could appear multiple times, use COUNT(DISTINCT entity) to avoid duplicate counting.
2. Always use case-insensitive comparison with toLower() function (not TOLOWER).
3. For keyword searches, be comprehensive by:
   - Using multiple CONTAINS clauses to catch word variations
   - Combining them with OR operators
   - Example: WHERE (toLower(c.name) CONTAINS 'art' OR toLower(c.name) CONTAINS 'exhibition' OR toLower(c.name) CONTAINS 'gallery')
   - Checking all Project/Event/Tag/Category for event types
4. When appropriate, use regular expressions for more flexible matching:
   - Example: WHERE e.name =~ '(?i).*(art|exhibition|gallery).*'
5. IMPORTANT: Events have these relationships (NEVER use any other relationships not listed here):
   - (Event)-[:HAS_TOPIC]->(Tag)
   - (Event)-[:BELONGS_TO]->(Category)
   - (Event)-[:TAKES_PLACE_IN]->(Location)
   - (Event)-[:PART_OF]->(Project)
   - (Coordinator)-[:COORDINATES]->(Event)
   - (Guest)-[:PARTICIPATES_IN]->(Event)
6. IMPORTANT: For people participating in events, use the Guest label, NOT Person:
   - CORRECT: MATCH (g:Guest)-[:PARTICIPATES_IN]->(e:Event)
   - INCORRECT: MATCH (p:Person)-[:PARTICIPATES_IN]->(e:Event)
7. When handling participant exclusions, use proper NOT EXISTS pattern with a full match clause inside.
8. For COUNT queries (when the user asks "how many"), just return the count without collecting IDs:
   - Use: RETURN COUNT(DISTINCT e) AS eventCount
   - This is more efficient for large result sets
9. Always return the event ID (e.id) in your results to aid in verification.

Here are examples of well-formed Cypher queries:

Example 1: Find all music events with more than 50 participants
```cypher
MATCH (e:Event)-[:BELONGS_TO]->(c:Category)
WHERE (toLower(c.name) CONTAINS 'music' OR toLower(c.name) CONTAINS 'concert' OR toLower(c.name) CONTAINS 'performance') 
  AND e.number_of_participants > 50
RETURN e.name AS eventName, e.id AS eventId, e.number_of_participants AS participants
ORDER BY e.number_of_participants DESC
```

Example 2: Find events coordinated by John Smith
```cypher
MATCH (c:Coordinator)-[:COORDINATES]->(e:Event)
WHERE toLower(c.name) = 'john smith'
RETURN e.name AS eventName, e.id AS eventId, c.name AS coordinator
```

Example 3: Find art events with multiple coordinators
```cypher
MATCH (e:Event)-[:BELONGS_TO]->(cat:Category)
WHERE toLower(cat.name) CONTAINS 'art' OR toLower(cat.name) CONTAINS 'exhibition' OR toLower(cat.name) CONTAINS 'gallery'
WITH e
MATCH (c:Coordinator)-[:COORDINATES]->(e)
WITH e, COUNT(DISTINCT c) AS coordinatorCount
WHERE coordinatorCount > 1
RETURN e.name AS eventName, e.id AS eventId, coordinatorCount
ORDER BY coordinatorCount DESC
```

Example 4: Find events where Alice participated but Bob did not
```cypher
MATCH (g1:Guest)-[:PARTICIPATES_IN]->(e:Event)
WHERE toLower(g1.name) = 'alice'
AND NOT EXISTS {{
  MATCH (g2:Guest)-[:PARTICIPATES_IN]->(e)
  WHERE toLower(g2.name) = 'bob'
}}
RETURN e.name AS eventName, e.id AS eventId
```

Example 5: Find events related to both jazz and improvisation
```cypher
MATCH (e:Event)
MATCH (e)-[:HAS_TOPIC]->(t:Tag)
WHERE toLower(t.name) CONTAINS 'jazz' OR toLower(t.name) CONTAINS 'improvisation'
WITH e
MATCH (e)-[:BELONGS_TO]->(c:Category)
WHERE toLower(c.name) CONTAINS 'jazz' OR toLower(c.name) CONTAINS 'improvisation'
RETURN e.name AS eventName, e.id AS eventId
```

Example 6: Count events on a specific date (9/24/2021)
```cypher
MATCH (e:Event)
WHERE e.start_date = '2021-09-24'
RETURN COUNT(e) AS eventCount
```

Example 7: Count events in a specific month (September 2021)
```cypher
MATCH (e:Event)
WHERE e.start_date_year_month = '2021-09'
RETURN COUNT(e) AS eventCount
```

Example 8: Count evening events (starting at or after 6 PM)
```cypher
MATCH (e:Event)
WHERE e.start_time >= '18:00'
RETURN COUNT(e) AS eventCount
```

Example 9: Find events happening in any August month
```cypher
MATCH (e:Event)
WHERE e.start_date CONTAINS '-08-'
RETURN e.id AS eventId, e.name AS eventName, e.start_date AS date
ORDER BY e.start_date
```

Example 10: Find outdoor events (since there's no outdoor property)
```cypher
MATCH (e:Event)-[:TAKES_PLACE_IN]->(l:Location)
WHERE toLower(l.name) CONTAINS 'park' OR 
      toLower(l.name) CONTAINS 'outdoor' OR 
      toLower(l.name) CONTAINS 'outside' OR 
      toLower(l.name) CONTAINS 'open-air' 
RETURN e.name AS eventName, e.id AS eventId, e.start_date AS date, l.name AS location
ORDER BY e.start_date
```

Example 11: Find events where a particular guest participated alone
```cypher
MATCH (g:Guest)-[:PARTICIPATES_IN]->(e:Event)
WHERE toLower(g.name) = 'monika borycka'
WITH e
MATCH (participant:Guest)-[:PARTICIPATES_IN]->(e)
WITH e, COUNT(participant) AS participantCount
WHERE participantCount = 1
RETURN e.name AS eventName, e.id AS eventId, e.start_date AS date
ORDER BY e.start_date DESC
```

Example 12: Find jazz events by searching in Categories, Tags, and Event names with multiple conditions
```cypher
MATCH (e:Event)
WHERE e.number_of_participants > 50
  AND e.start_date > '2021-02-28'
  AND (
    toLower(e.name) CONTAINS 'jazz'
    OR EXISTS {{
      MATCH (e)-[:BELONGS_TO]->(c:Category)
      WHERE toLower(c.name) CONTAINS 'jazz'
    }}
    OR EXISTS {{
      MATCH (e)-[:HAS_TOPIC]->(t:Tag)
      WHERE toLower(t.name) CONTAINS 'jazz'
    }}
  )
RETURN e.name AS eventName, e.id AS eventId, e.number_of_participants AS participants
ORDER BY e.start_date
```

IMPORTANT: Format your response as follows:
1. First provide a brief explanation of your approach
2. Then provide ONLY the Cypher query enclosed in triple backticks like:
```cypher
YOUR QUERY HERE
```
3. Do not include any explanatory text within the triple backticks

Cypher Query:
"""

# QA prompt template
QA_TEMPLATE = """
You are an expert in analyzing Neo4j graph database query results for cultural events.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}

The user has asked: {question}

Based on the Cypher query results from the Neo4j database, provide a detailed and accurate answer.

Context from database query results:
Graph Results:
{graph_results}

Vector Search Context (additional information):
{vector_results}

Important guidelines:
1. If the results are empty, explain that no matching events were found.
2. If there are specific counts or statistics in the query results, include them in your answer.
3. Format lists of events in a readable way.
4. If the question asked "how many", make sure to count the results and provide the number.
5. Always answer based purely on the data in the query results, not on general knowledge.
6. When returning lists of events, format them in a clear, structured way.
7. For events with dates, format them in a human-readable way.
8. For events with locations, include the location information in your answer.
9. For events with coordinators or guests, mention them when relevant.
10. Prioritize information from the Graph Results, and use Vector Search Context as supplementary information.
11. IMPORTANT: If the user query includes specific filtering criteria (like "more than X participants", "outdoor events", etc.), make sure to apply these filters to both Graph Results and Vector Search Context. For example, if the user asks for "events with more than 100 participants", check the number_of_participants field in the metadata and only include events that meet this criterion.
12. When filtering based on numeric values (like number_of_participants), make sure to convert string values to numbers before comparing.
13. Include the event IDs in your response when listing events, as these are useful for testing and verification.

Based solely on the provided database results, answer the user's question:
"""

# Create prompt templates
cypher_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "query"],
)

qa_prompt = PromptTemplate(
    template=QA_TEMPLATE,
    input_variables=["schema", "question", "graph_results", "vector_results"],
)

# Query expansion prompt template
QUERY_EXPANSION_TEMPLATE = """Extract and expand the key search terms from this query. 
Focus on the most important semantic concepts and add related terms that would help in a vector search.

FILTER OUT common words like:
- Articles (the, a, an)
- Prepositions (in, on, at, to, from, by, of)
- Conjunctions (and, or, but)
- Common verbs (find, show, list, display, return, give)
- Filler words (all, with, more, than, about, for)

KEEP important semantic terms related to:
- Event types (concert, exhibition, workshop, festival, conference)
- Art forms (music, visual art, theater, dance)
- Specific genres (jazz, classical, rock, contemporary)
- Locations and names

For example, if the query mentions 'jazz concert', you might add terms like 'music', 'performance', 'band', etc.

ONLY return the expanded search terms as a space-separated list of words. Do not include any explanations or other text.

Query: {query}

Expanded search terms: """

query_expansion_prompt = PromptTemplate(
    template=QUERY_EXPANSION_TEMPLATE,
    input_variables=["query"],
)
