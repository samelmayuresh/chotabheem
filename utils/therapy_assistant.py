# utils/therapy_assistant.py - Advanced AI therapy assistant with context awareness
import streamlit as st
import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

@dataclass
class TherapySession:
    """Track therapy session context"""
    user_id: str
    session_start: datetime
    emotions_history: List[Dict]
    conversation_history: List[Dict]
    therapeutic_goals: List[str]
    current_mood_trend: str

class TherapyAssistant:
    """Advanced AI therapy assistant with context awareness and specialized techniques"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.session_context = {}
        
        # Therapeutic techniques database
        self.techniques = {
            "cbt": {
                "name": "Cognitive Behavioral Therapy",
                "triggers": ["anxiety", "depression", "negative_thoughts"],
                "approach": "Challenge negative thought patterns and develop coping strategies"
            },
            "mindfulness": {
                "name": "Mindfulness-Based Therapy",
                "triggers": ["stress", "overwhelm", "racing_thoughts"],
                "approach": "Focus on present moment awareness and acceptance"
            },
            "dbt": {
                "name": "Dialectical Behavior Therapy",
                "triggers": ["emotional_regulation", "interpersonal_issues"],
                "approach": "Build distress tolerance and emotional regulation skills"
            },
            "solution_focused": {
                "name": "Solution-Focused Brief Therapy",
                "triggers": ["goal_setting", "problem_solving"],
                "approach": "Focus on solutions and strengths rather than problems"
            }
        }
    
    def get_therapeutic_response(self, user_input: str, emotion_data: Dict, 
                               session_id: str = "default") -> Dict:
        """Generate contextual therapeutic response"""
        
        # Update session context
        self._update_session_context(session_id, user_input, emotion_data)
        
        # Select appropriate therapeutic approach
        technique = self._select_technique(emotion_data)
        
        # Generate response based on context and technique
        response = self._generate_ai_response(user_input, emotion_data, technique, session_id)
        
        # Add therapeutic exercises if appropriate
        exercises = self._suggest_exercises(emotion_data, technique)
        
        return {
            "response": response,
            "technique": technique,
            "exercises": exercises,
            "session_insights": self._get_session_insights(session_id)
        }
    
    def _update_session_context(self, session_id: str, user_input: str, emotion_data: Dict):
        """Update session context with new interaction"""
        if session_id not in self.session_context:
            self.session_context[session_id] = {
                "start_time": datetime.now(),
                "interactions": [],
                "emotion_history": [],
                "identified_patterns": [],
                "therapeutic_progress": []
            }
        
        session = self.session_context[session_id]
        session["interactions"].append({
            "timestamp": datetime.now(),
            "user_input": user_input,
            "emotion": emotion_data
        })
        session["emotion_history"].append(emotion_data)
        
        # Analyze patterns if we have enough data
        if len(session["interactions"]) >= 3:
            self._analyze_patterns(session_id)
    
    def _select_technique(self, emotion_data: Dict) -> str:
        """Select most appropriate therapeutic technique"""
        primary_emotion = emotion_data.get("primary", "neutral")
        confidence = emotion_data.get("confidence", 0.0)
        
        # Map emotions to techniques
        technique_mapping = {
            "anxiety": "cbt",
            "fear": "cbt", 
            "nervousness": "mindfulness",
            "stress": "mindfulness",
            "anger": "dbt",
            "sadness": "cbt",
            "depression": "cbt",
            "confusion": "solution_focused",
            "overwhelm": "mindfulness"
        }
        
        return technique_mapping.get(primary_emotion, "mindfulness")
    
    def _generate_ai_response(self, user_input: str, emotion_data: Dict, 
                            technique: str, session_id: str) -> str:
        """Generate AI response using selected therapeutic technique"""
        
        # Build context-aware prompt
        session = self.session_context.get(session_id, {})
        recent_emotions = [e.get("primary", "neutral") for e in session.get("emotion_history", [])[-3:]]
        
        system_prompt = self._build_system_prompt(technique, recent_emotions)
        user_prompt = self._build_user_prompt(user_input, emotion_data, session)
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7
                },
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logging.error(f"AI API error: {response.status_code}")
                return self._get_fallback_response(emotion_data["primary"])
                
        except Exception as e:
            logging.error(f"AI response generation failed: {e}")
            return self._get_fallback_response(emotion_data["primary"])
    
    def _build_system_prompt(self, technique: str, recent_emotions: List[str]) -> str:
        """Build therapeutic system prompt"""
        technique_info = self.techniques.get(technique, self.techniques["mindfulness"])
        
        return f"""You are a compassionate AI therapist specializing in {technique_info['name']}.
        
        Approach: {technique_info['approach']}
        
        Recent emotional pattern: {' → '.join(recent_emotions) if recent_emotions else 'First interaction'}
        
        Guidelines:
        - Be warm, empathetic, and non-judgmental
        - Use evidence-based therapeutic techniques
        - Provide practical, actionable advice
        - Validate emotions while encouraging growth
        - Keep responses concise but meaningful (150-200 words)
        - Ask thoughtful follow-up questions when appropriate
        - Never provide medical advice or diagnose conditions"""
    
    def _build_user_prompt(self, user_input: str, emotion_data: Dict, session: Dict) -> str:
        """Build context-aware user prompt"""
        interaction_count = len(session.get("interactions", []))
        
        prompt = f"""Current user message: "{user_input}"

        Detected emotion: {emotion_data.get('primary', 'neutral')} (confidence: {emotion_data.get('confidence', 0):.1%})
        
        Session context: This is interaction #{interaction_count + 1}"""
        
        if interaction_count > 0:
            prompt += f"\nPrevious themes: {', '.join(session.get('identified_patterns', []))}"
        
        return prompt
    
    def _suggest_exercises(self, emotion_data: Dict, technique: str) -> List[Dict]:
        """Suggest therapeutic exercises based on emotion and technique"""
        primary_emotion = emotion_data.get("primary", "neutral")
        
        exercises = {
            "anxiety": [
                {
                    "name": "5-4-3-2-1 Grounding",
                    "description": "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste",
                    "duration": "2-3 minutes"
                },
                {
                    "name": "Box Breathing",
                    "description": "Breathe in for 4, hold for 4, out for 4, hold for 4",
                    "duration": "5 minutes"
                }
            ],
            "sadness": [
                {
                    "name": "Gratitude Journal",
                    "description": "Write down 3 things you're grateful for today",
                    "duration": "5 minutes"
                },
                {
                    "name": "Self-Compassion Break",
                    "description": "Acknowledge your pain, remember you're not alone, offer yourself kindness",
                    "duration": "3 minutes"
                }
            ],
            "anger": [
                {
                    "name": "Progressive Muscle Relaxation",
                    "description": "Tense and release each muscle group from toes to head",
                    "duration": "10 minutes"
                },
                {
                    "name": "Thought Record",
                    "description": "Write down the triggering thought and challenge its accuracy",
                    "duration": "5 minutes"
                }
            ],
            "stress": [
                {
                    "name": "Mindful Walking",
                    "description": "Take a slow walk focusing on each step and your surroundings",
                    "duration": "10 minutes"
                },
                {
                    "name": "Body Scan",
                    "description": "Mentally scan your body from head to toe, noticing tension",
                    "duration": "8 minutes"
                }
            ]
        }
        
        return exercises.get(primary_emotion, exercises["stress"])
    
    def _analyze_patterns(self, session_id: str):
        """Analyze emotional and conversational patterns"""
        session = self.session_context[session_id]
        emotions = [e.get("primary", "neutral") for e in session["emotion_history"]]
        
        # Identify recurring emotions
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Find patterns
        patterns = []
        if emotion_counts:
            most_common = max(emotion_counts, key=emotion_counts.get)
            if emotion_counts[most_common] >= 2:
                patterns.append(f"Recurring {most_common} emotions")
        
        # Trend analysis
        if len(emotions) >= 3:
            recent_trend = emotions[-3:]
            if all(e in ["sadness", "anxiety", "fear"] for e in recent_trend):
                patterns.append("Negative emotional trend")
            elif all(e in ["joy", "gratitude", "optimism"] for e in recent_trend):
                patterns.append("Positive emotional trend")
        
        session["identified_patterns"] = patterns
    
    def _get_session_insights(self, session_id: str) -> Dict:
        """Generate insights about the therapy session"""
        session = self.session_context.get(session_id, {})
        
        if not session:
            return {"insights": [], "recommendations": []}
        
        insights = []
        recommendations = []
        
        # Session duration insight
        if "start_time" in session:
            duration = datetime.now() - session["start_time"]
            if duration.total_seconds() > 1800:  # 30 minutes
                insights.append("Extended therapy session - showing commitment to self-care")
        
        # Emotional progress
        emotions = session.get("emotion_history", [])
        if len(emotions) >= 2:
            first_emotion = emotions[0].get("primary", "neutral")
            latest_emotion = emotions[-1].get("primary", "neutral")
            
            positive_emotions = ["joy", "gratitude", "optimism", "relief"]
            if first_emotion not in positive_emotions and latest_emotion in positive_emotions:
                insights.append("Positive emotional shift during session")
                recommendations.append("Continue with current coping strategies")
        
        # Pattern-based recommendations
        patterns = session.get("identified_patterns", [])
        if "Recurring anxiety emotions" in patterns:
            recommendations.append("Consider regular mindfulness practice")
        if "Negative emotional trend" in patterns:
            recommendations.append("Focus on self-compassion exercises")
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "session_quality": self._assess_session_quality(session)
        }
    
    def _assess_session_quality(self, session: Dict) -> str:
        """Assess the quality of the therapy session"""
        interaction_count = len(session.get("interactions", []))
        patterns = len(session.get("identified_patterns", []))
        
        if interaction_count >= 5 and patterns > 0:
            return "High engagement - productive session"
        elif interaction_count >= 3:
            return "Good engagement - meaningful conversation"
        else:
            return "Initial exploration - building rapport"
    
    def _get_fallback_response(self, emotion: str) -> str:
        """Fallback responses when AI is unavailable"""
        fallbacks = {
            "anxiety": "I understand you're feeling anxious. Take a deep breath with me. Anxiety is temporary, and you have the strength to work through this.",
            "sadness": "I hear that you're feeling sad, and that's completely valid. These feelings are part of being human. What's one small thing that usually brings you comfort?",
            "anger": "I can sense your anger, and it's okay to feel this way. Let's work together to understand what's behind this feeling and find healthy ways to express it.",
            "fear": "Fear can feel overwhelming, but you're safe right now. Let's ground ourselves in the present moment. What are three things you can see around you?",
            "stress": "Stress is your body's way of responding to challenges. You're doing better than you think. What's one thing you can let go of right now?"
        }
        
        return fallbacks.get(emotion, "I'm here to listen and support you. Your feelings are valid, and together we can work through whatever you're experiencing.")