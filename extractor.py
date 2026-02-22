import openai
from typing import List, Dict

class TripleExtractor:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.instruction = """
Your task is to help me extract subject-relation-object triples from sentences.
(1) Identify the main subject, the relation (verb or action), and the object.
(2) Format exactly as: "Subject: XXXX; Relation: XXXX; Object: XXXX"
(3) Start the answer line with "####".
"""
        # Add your few-shot demonstrations here
        self.demos = [
            {"text": "A black husband would satisfy that brides maid behind his new wife's back.", 
             "triple": "Subject: black husband; Relation: satisfy; Object: brides maid"}
        ]

    def extract_triples(self, dataset: List[Dict]) -> List[Dict]:
        """Uses an LLM to convert biased text into SRO triples."""
        print("Extracting SRO triples using LLM API...")
        for item in dataset:
            prompt = self._build_prompt(item["text"])
            
            # Call to LLM API (Replace with specific SDK logic if using Gemini)
            response = openai.ChatCompletion.create(
                model="gpt-4o", # or chosen model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            raw_output = response.choices[0].message.content
            item["extracted_triple"] = self._parse_output(raw_output)
            
        return dataset

    def _build_prompt(self, target_text: str) -> str:
        prompt = self.instruction + "\n\n"
        for demo in self.demos:
            prompt += f"Problem:\n{demo['text']}\nAnswer:\n#### {demo['triple']}\n\n"
        prompt += f"Problem:\n{target_text}\nAnswer:\n"
        return prompt

    def _parse_output(self, output: str) -> Dict[str, str]:
        """Parses 'Subject: X; Relation: Y; Object: Z' into a dictionary."""
        # Clean and parse the string here
        try:
            core_text = output.split("####")[1].strip()
            parts = core_text.split(";")
            return {
                "subject": parts[0].split(":")[1].strip(),
                "relation": parts[1].split(":")[1].strip(),
                "object": parts[2].split(":")[1].strip(),
                "target_new": "none" # Standard BTBR placeholder
            }
        except Exception as e:
            return {"subject": "", "relation": "", "object": "", "target_new": "none"}