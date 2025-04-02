# utils/prompt_loader.py
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

class PromptLoader:
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(Path(__file__).parent.parent / 'prompt'),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def get_prompt(
        self, 
        context: dict,
        category: str
    ) -> str:
        prompt = f"{category}.jinja"
        try:
            template = self.env.get_template(f"{category}.jinja")
        except TemplateNotFound:
            template = self.env.get_template("one_sentence_novel.jinja")
        return template.render(context)