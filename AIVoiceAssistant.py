import re
import logging
import os
from pipeline import RAGPipeline

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AIVoiceAssistant:
    def __init__(self, pdf_path="./knowledge_base/knowledge_base_two.pdf"):
        self.rag = RAGPipeline(
            llm_model="llama3.2:1b",
            embedding_model="nomic-embed-text",
            temperature=0.7,
            max_tokens=500,
        )
        try:
            self.rag.load_state()
            logging.info("Loaded existing knowledge base")
        except Exception as e:
            logging.error(f"Error loading knowledge base: {e}")
            logging.info("Creating new knowledge base...")
            self.rag.process_document(pdf_path)
            self.rag.save_state()
            logging.info("Knowledge base created successfully!")

        # Define greeting and emergency keywords
        self.greeting_patterns = [
            r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b",
        ]

        self.emergency_keywords = [
            "heart attack",
            "choking",
            "difficulty breathing",
            "unconscious",
            "severe bleeding",
            "stroke",
            "seizure",
            "anaphylaxis",
            "allergic reaction",
        ]

    def is_greeting(self, text: str) -> bool:
        """Check if the input text contains only a greeting."""
        text = text.lower().strip()
        return any(re.fullmatch(pattern, text) for pattern in self.greeting_patterns)

    def is_emergency(self, text: str) -> bool:
        """Check if the input text contains an emergency situation."""
        text = text.lower().strip()
        return any(keyword in text for keyword in self.emergency_keywords)

    def get_emergency_response(self, text: str) -> str:
        """Return an emergency response based on detected keywords."""
        emergency_responses = {
            "heart attack": "🚨 If you suspect a heart attack, call emergency services immediately. Chew and swallow an aspirin unless allergic. Keep the person calm and seated.",
            "choking": "🚨 For choking, encourage coughing if the person can still breathe. If they cannot, perform the Heimlich maneuver immediately.",
            "difficulty breathing": "🚨 Difficulty breathing is a medical emergency. Loosen tight clothing, sit the person upright, and call emergency services immediately.",
            "unconscious": "🚨 If someone is unconscious, check for breathing. If absent, start CPR and call emergency services immediately.",
            "severe bleeding": "🚨 Apply direct pressure with a clean cloth to stop bleeding. If the bleeding does not stop, seek emergency help immediately.",
            "stroke": "🚨 If you suspect a stroke, act FAST: Face drooping, Arm weakness, Speech difficulty. Call emergency services immediately.",
            "seizure": "🚨 Keep the person safe during a seizure by clearing the area and placing them on their side. Do not restrain them. Seek medical help.",
            "anaphylaxis": "🚨 Anaphylaxis is life-threatening. Use an epinephrine injector (EpiPen) if available and call emergency services immediately.",
            "allergic reaction": "🚨 If an allergic reaction is mild, give an antihistamine. If breathing is affected, use an EpiPen and call emergency services.",
        }

        for keyword, response in emergency_responses.items():
            if keyword in text:
                return response
        return "🚨 This sounds like a medical emergency. Please seek immediate professional help."

    def interact_with_llm(self, customer_query: str) -> str:
        """Process customer query and return AI response."""
        try:
            # Check for emergency situations first
            if self.is_emergency(customer_query):
                return self.get_emergency_response(customer_query)

            # If the user says only a greeting, respond with a predefined message
            if self.is_greeting(customer_query):
                return "Hello! I'm your AI healthcare assistant. How can I assist you today? 😊"

            # Otherwise, let AI generate the response for the full query
            enhanced_query = self._enhance_query(customer_query)
            result = self.rag.query(enhanced_query)

            return result["response"]

        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return "I apologize, but I'm having trouble processing your request. Could you please repeat that?"

    def _enhance_query(self, query: str) -> str:
        """Enhance query for better AI processing."""

        context = """
        You are an AI healthcare assistant that provides direct, factual responses to medical queries.
        
        Guidelines:
        1. Answer the exact question asked - do not make assumptions
        2. Keep responses clear and concise
        3. Use simple, natural language
        4. Only provide medical facts from verified sources
        5. Include safety disclaimers when appropriate
        
        Important:
        - Do not reference examples or hypothetical scenarios
        - Do not make assumptions about symptoms or conditions not mentioned
        - Stay focused on the specific question asked
        - If the query is unclear, ask for clarification
        - For general questions about capabilities, explain what you can actually do
        
        For example, if someone asks "How can you help me?", explain your actual capabilities:
        - Answering medical questions using verified information
        - Providing general health guidance
        - Explaining medical terms and procedures
        - Identifying emergency situations
        - Suggesting when to seek professional care

        Now, process this query:
        """

        return context + query
