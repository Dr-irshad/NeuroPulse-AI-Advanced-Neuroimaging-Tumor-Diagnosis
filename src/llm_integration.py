"""
llm_integration.py
------------------
Integrates the DeepSeek-R1 model (via Ollama + LangChain) for clinical reasoning.
This script demonstrates how structured data from the segmentation stage
is passed to the LLM for diagnostic text generation.
"""

from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class DiagnosticReasoner:
    """
    Uses an LLM to generate a clinically formatted pre-diagnosis report
    based on structured tumor data from YOLO segmentation.
    """

    def __init__(self, model_name: str = "deepseek-r1"):
        self.llm = Ollama(model=model_name)
        self.template = PromptTemplate(
            input_variables=["tumor_class", "confidence", "area"],
            template=(
                "You are a medical AI assistant analyzing MRI tumor segmentation results.\n"
                "Detected tumor type: {tumor_class}\n"
                "Confidence: {confidence}\n"
                "Approximate area (pixels): {area}\n\n"
                "Generate a short diagnostic summary, "
                "including probable grade, anatomical location, and follow-up recommendations."
            ),
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.template)
        print(f"[INFO] Initialized DeepSeek-R1 LLM for reasoning ({model_name})")

    def generate_report(self, tumor_data: dict) -> str:
        """
        Generate a natural-language report using the LLM.
        """
        response = self.chain.run(
            tumor_class=tumor_data["tumor_class"],
            confidence=f"{tumor_data['mean_confidence']:.2f}",
            area=int(tumor_data["tumor_area_pixels"]),
        )
        print("[INFO] Generated diagnostic report via LLM.")
        return response.strip()

# Example usage
if __name__ == "__main__":
    reasoner = DiagnosticReasoner()
    sample_input = {
        "tumor_class": "glioma",
        "mean_confidence": 0.94,
        "tumor_area_pixels": 12834
    }
    report = reasoner.generate_report(sample_input)
    print(report)

