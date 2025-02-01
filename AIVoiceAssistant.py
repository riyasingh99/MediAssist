import re
import logging
import os
from pipeline import RAGPipeline

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AIVoiceAssistant:
    def __init__(self, pdf_path="./B5084.pdf"):
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

        # Initial context for health-related queries
        context = """
        You are an AI healthcare assistant trained to provide accurate medical information without any empathy. 
        Focus on providing direct, medically accurate answers to user queries, especially for weight gain, 
        nutrition, and fitness.

        Guidelines:
        1. Provide direct, concise answers to the user's query.
        2. For health-related queries, include necessary precautions, warnings, and next steps.
        3. Do not offer vague responses. Always aim to be specific and actionable.
        4. Ensure to guide users on what they should do next, such as consulting a professional if needed.
        5. Do not give any definitive diagnosis, always mention that a healthcare provider should be consulted for any serious concerns.
        6. If the query is unclear, ask the user to clarify before providing an answer.
        7. Always back the responses with general medical knowledge (e.g., guidelines, common treatments).

        Format for responses:
        1. **Direct answer**: Provide a clear, precise answer to the user's query.
        2. **Precautions or warnings**: If needed, provide any necessary warnings or precautions.
        3. **Next steps or recommendations**: Offer any next steps, additional recommendations, or where the user can seek further help.

        Example response for weight gain query:

        1. **User Input**: "How can I gain weight as a gym-goer?"
        **Response**:
            - **Direct answer**: "To gain weight in a healthy manner, focus on consuming a calorie surplus through a nutrient-dense diet. This means eating more than you burn, with an emphasis on lean proteins, healthy fats, and complex carbohydrates."
            - **Precautions or warnings**: "Avoid processed, high-sugar foods, as they can lead to unhealthy fat gain and potential metabolic issues. Focus on whole foods for the best results."
            - **Next steps or recommendations**: 
                - "Increase your calorie intake by adding healthy, calorie-dense foods like nuts, avocados, quinoa, and whole grains."
                - "Incorporate resistance training in your gym routine to build muscle mass, not just fat. Compound exercises like squats, deadlifts, and bench presses are highly effective for muscle growth."
                - "Aim for a daily caloric surplus of about 300-500 calories, and track your progress to ensure you’re gaining lean mass rather than fat."
                - "Consider consulting a dietitian to personalize your nutrition plan."

        Now, process the current query:
        """

        return context + query
