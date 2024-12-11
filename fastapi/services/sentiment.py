from fastapi import  Response, HTTPException
import requests
import json

async def ask_question(prompt: str) -> Response:
    ollama_url = "http://ollama:11434/api/generate"

    # Defining the prompt with additional instructions
    full_prompt = (
        f"{prompt}\n\n"
        "The above text is the input on which you need to perform the following tasks as the data scientist of s2T.ai:\n"
        "1. **Entity Recognition:** Extract relevant entities such as names, locations, organizations, dates, companies, and any other significant entities present in the text.\n"
        "2. **Relationship Extraction:** Identify and structure meaningful relationships between the extracted entities, such as roles, associations, or contextual links.\n"
        "3. **Sentiment Analysis:** Analyze the sentiment of the text (e.g., positive, negative, neutral) and include any key highlights or insights related to the sentiment.\n\n"
        "Return the output in a well-structured JSON format, clearly separating entities, relationships, and sentiment analysis results.\n"
        "If the input text is invalid or no relevant information is found, return an empty JSON object."
        "Return the response as an array of JSON objects. Don't add any descriptive text around the resulted array of objects."
    )

    # Creating the payload for the POST request
    payload = {
        "model": "phi3",
        "prompt": full_prompt,
        "stream": False
    }
    try:
        # Sending the request to the Ollama API
        response = requests.post(ollama_url, json=payload)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error from Ollama API: {response.text}"
            )

        # Parsing the API response
        data = json.loads(response.text)
        print(data)
        # if response includes markdown syntax
        if data['response'].startswith("```json") and data['response'].endswith("```"):
            # Remove the markdown syntax
            cleaned_response = data['response'].strip('```json').strip('```')
        else:
            # If the response is already in json format without markdown syntax
            cleaned_response = data['response']

        print(cleaned_response)

        return Response(content=json.dumps(cleaned_response, indent=2), media_type="application/json")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing JSON response: {str(e)}")
