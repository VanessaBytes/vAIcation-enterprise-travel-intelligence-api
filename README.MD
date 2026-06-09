# Your Task
We've built an agent (python file attached) that demonstrates how the Tavily API can be used in an agentic system. The system has limitations, and your task is to create a meaningful improvement that demonstrates either an interesting technical approach and/or creates clear business value.

We want to see your creativity, how you think, knowledge of AI and your specific skills in action!

**Example improvements:** retrieval enhancements, RL, domain-specific use case, integrations, context engineering innovation, evaluation framework, etc...

**Deliverables:**

- GitHub repository with your implementation
- Brief technical statement explaining your approach, thought process, and the value it creates (technical and/or business perspective)


## Setup

1. **Install dependencies:**
```bash
python3 -m venv venv or python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Set up environment variables:**
Create a `.env` file in the project root with:
```
TAVILY_API_KEY=your_tavily_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

```bash
python app.py.py
```

The API will start on `http://localhost:8000`

## Usage

Send a POST request to `/run` with a query:

```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in AI?"}'
```