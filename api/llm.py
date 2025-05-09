from api.prompt import Prompt
import os
import pandas as pd
from openai import OpenAI
import pyimgur
from tenacity import retry, wait_exponential, stop_after_attempt

class ChatGPT:
    """
    A class for generating responses using OpenAI's Responses API with a custom model,
    optional CSV context injection, and file search tool integration.
    """

    def __init__(self,
                 model_name: str = None,
                 temperature: float = None,
                 max_tokens: int = None,
                 data_path: str = None,
                 vector_store_id: str = None):
        # configure model parameters
        self.model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(temperature or os.getenv("OPENAI_TEMPERATURE", 0))
        self.max_tokens = int(max_tokens or os.getenv("OPENAI_MAX_TOKENS", 600))

        # init client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

        # optional CSV data context
        self.data = None
        if data_path:
            self.load_data(data_path)

        # optional file-search tool vector store
        self.vector_store_id = vector_store_id

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load CSV or Excel into DataFrame for context injection.
        """
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported data format: must be .csv or .xlsx")
        self.data = df
        return df

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def get_response(self, user_messages: list[dict]) -> str:
        """
        Generate a response via Responses API. Automatically injects CSV context
        and configures file_search tool if vector_store_id is set.
        user_messages: list of {"role":..., "content":...} entries
        """
        # build input list
        inputs = []
        # inject data context if available
        if self.data is not None:
            context_md = self.data.head(5).to_markdown(index=False)
            inputs.append({
                "role": "system",
                "content": [{"type": "input_text", "text": f"Here is the data context:\n{context_md}"}]
            })
        # append user conversation
        for msg in user_messages:
            inputs.append({
                "role": msg["role"],
                "content": [{"type": "input_text", "text": msg["content"]}]
            })

        # prepare kwargs
        req = {
            "model": self.model,
            "input": inputs,
            "text": {"format": {"type": "text"}},
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "store": True
        }
        # attach file_search tool if provided
        if self.vector_store_id:
            req["tools"] = [{"type": "file_search", "vector_store_ids": [self.vector_store_id]}]

        response = self.client.responses.create(**req)
        # extract assistant's last output_text
        for choice in response.choices:
            for content in choice.message.content:
                if content.get("type") == "output_text":
                    return content.get("text")
        return ""

    def add_user_message(self, text: str) -> dict:
        """
        Wrap a user string into the expected message dict.
        """
        return {"role": "user", "content": text}

    def process_image_link(self, image_url: str) -> str:
        """
        Analyze image via Responses API chat endpoint.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Analyze the content of this image."},
                {"role": "user", "content": image_url}
            ],
            temperature=self.temperature,
            max_tokens=100
        )
        return response.choices[0].message.content

    def get_user_image(self, image_content) -> str:
        path = './static/temp.png'
        with open(path, 'wb') as fd:
            for chunk in image_content.iter_content():
                fd.write(chunk)
        return path

    def upload_img_link(self, imgpath: str) -> str:
        IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")
        if not IMGUR_CLIENT_ID:
            raise ValueError("Missing IMGUR_CLIENT_ID environment variable.")
        im = pyimgur.Imgur(IMGUR_CLIENT_ID)
        uploaded_image = im.upload_image(imgpath, title="Uploaded with PyImgur")
        return uploaded_image.link

# Usage example:
# bot = ChatGPT(model_name="ft:gpt-4o-mini-2024-07-18:test::BUP18udX", data_path="./data/context.csv", vector_store_id="vs_... )
# msgs = [bot.add_user_message("請根據第十屆理監事名單回答問題")] 
# print(bot.get_response(msgs))
