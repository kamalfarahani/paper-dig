import time
import requests

from pathlib import Path

from paper_dig.markdown_extractor import AbstractMarkdownExtractor


class DatalabMarkdownExtractor(AbstractMarkdownExtractor):
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://www.datalab.to/api/v1/marker",
        max_polls: int = 100,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.max_polls = max_polls

    def extract_markdown(self, file_path: Path) -> str:
        form_data = {
            "file": (
                "test.pdf",
                open(file_path, "rb"),
                "application/pdf",
            ),
            "langs": (None, "English"),
            "force_ocr": (None, False),
            "paginate": (None, True),
            "output_format": (None, "markdown"),
            "use_llm": (None, True),
            "strip_existing_ocr": (None, True),
            "disable_image_extraction": (None, True),
        }

        headers = {"X-Api-Key": self.api_key}

        response = requests.post(
            self.api_url,
            files=form_data,
            headers=headers,
        )

        data = response.json()
        check_url = data["request_check_url"]

        for i in range(self.max_polls):
            time.sleep(2)
            response = requests.get(
                check_url,
                headers=headers,
            )
            data = response.json()

            if data["status"] == "complete":
                break
        else:
            return ""

        return data["markdown"]
