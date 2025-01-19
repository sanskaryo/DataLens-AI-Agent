import json
from phi.model.groq import Groq
from phi.agent.duckdb import DuckDbAgent

from dotenv import load_dotenv
import os
load_dotenv()

data_analyst = DuckDbAgent(
    model=Groq(id="llama-3.3-70b-versatile"),
    # llama-3.3-70b-versatile
    markdown=True,
    semantic_model=json.dumps(
        {
            "tables": [
                {
                    "name": "movies",
                    "description": "Contains information about movies from IMDB.",
                    "path": "https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
                }
            ]
        },
        indent=2,
    ),
)

data_analyst.print_response(
    
    "Show me a histogram of ratings. "
    "Choose an appropriate bucket size but share how you chose it. "
    "Show me the result as a pretty ascii diagram",
    stream=True,
)