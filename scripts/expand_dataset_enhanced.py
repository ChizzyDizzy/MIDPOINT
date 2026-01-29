"""
SafeMind AI - Enhanced Dataset Expander
Generates larger, more diverse training datasets with lively, realistic responses.

IMPROVEMENTS OVER ORIGINAL:
- Support for 3000+ samples
- More diverse user inputs with natural variations
- Lively, empathetic responses that feel more human
- Better conversation flow patterns
- Multi-turn conversation support
- Slang and casual language variations
- More emotional range

USAGE:
  python3 expand_dataset_enhanced.py --output ../data/mental_health_dataset.json --num-samples 3000
"""

import json
import os
import random
import time
from typing import List, Dict, Tuple
import argparse


class EnhancedDatasetExpander:
    """Generate larger, more diverse, and more realistic training datasets"""

    def __init__(self, data_dir: str = "../data"):
        """Initialize with enhanced templates and variations"""
        self.data_dir = data_dir

        # Load existing template files
        self.crisis_patterns = self._load_json("crisis_patterns.json")
        self.cultural_templates = self._load_json("cultural_templates.json")
        self.response_templates = self._load_json("response_templates.json")

        # Enhanced user input patterns with natural variations
        self.user_inputs = self._create_enhanced_user_inputs()

        # Lively, empathetic response templates
        self.lively_responses = self._create_lively_responses()

        # Conversation starters and follow-ups
        self.conversation_patterns = self._create_conversation_patterns()

        print("Enhanced Dataset Expander initialized")
        print(f"  - {sum(len(v) for v in self.user_inputs.values())} unique input patterns")
        print(f"  - {sum(len(v) for v in self.lively_responses.values())} response templates")

    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file from data directory"""
        path = os.path.join(self.data_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found, using defaults")
            return {}

    def _create_enhanced_user_inputs(self) -> Dict[str, List[str]]:
        """Create diverse, natural user input patterns"""
        return {
            "anxiety": [
                "I've been feeling really anxious lately",
                "I can't stop worrying about everything",
                "My heart races and I feel panicky",
                "I'm constantly nervous and on edge",
                "I have panic attacks frequently",
                "I worry all the time and can't relax",
                "my anxiety is through the roof rn",
                "can't shake this anxious feeling",
                "everything makes me anxious these days",
                "i get so nervous i feel sick",
                "anxiety is ruining my life",
                "i panic over small things",
                "my mind won't stop racing",
                "i feel like something bad is going to happen",
                "the anxiety is overwhelming",
                "i'm scared all the time for no reason",
                "my chest gets tight when i'm stressed",
                "i can't breathe properly when anxious",
                "social situations make me so anxious",
                "i overthink everything",
                "what if something goes wrong?",
                "i can't control my worrying",
                "anxiety keeps me up at night",
                "i feel like i'm going crazy with worry",
            ],
            "depression": [
                "I feel sad all the time",
                "Nothing makes me happy anymore",
                "I have no energy to do anything",
                "I feel empty inside",
                "I've lost interest in everything I used to enjoy",
                "I feel hopeless about the future",
                "everything feels pointless",
                "i don't enjoy anything anymore",
                "i'm just so tired of everything",
                "why even bother getting out of bed",
                "life feels meaningless",
                "i feel numb all the time",
                "nothing brings me joy",
                "i've been crying a lot lately",
                "i feel like a failure",
                "everything is so hard",
                "i can't see things getting better",
                "i feel broken inside",
                "the sadness won't go away",
                "i'm exhausted but can't sleep",
                "i don't care about anything anymore",
                "i feel so empty",
                "what's the point of anything",
                "i'm tired of pretending to be okay",
            ],
            "stress": [
                "I'm under so much pressure",
                "Everything feels overwhelming",
                "I can't handle all this stress",
                "I'm so stressed I can't think straight",
                "The stress is affecting my health",
                "I feel like I'm drowning in responsibilities",
                "there's too much on my plate",
                "i can't catch a break",
                "stress is eating me alive",
                "i'm at my breaking point",
                "everything is piling up",
                "i feel like i'm going to snap",
                "the pressure is too much",
                "i can't keep up with everything",
                "i'm burned out",
                "stress is making me sick",
                "i have no time for myself",
                "i feel stretched too thin",
                "everyone wants something from me",
                "i'm running on empty",
            ],
            "academic_stress": [
                "I'm stressed about my A/L exams",
                "I failed my university exam",
                "The pressure to get high marks is too much",
                "I can't handle the academic workload",
                "I'm afraid I'll fail and disappoint everyone",
                "University admissions are making me anxious",
                "My exam results are causing me so much stress",
                "i bombed my exam",
                "my grades are slipping",
                "everyone expects me to get top marks",
                "i'm not smart enough for this",
                "the competition is so intense",
                "i studied so hard but still failed",
                "my future depends on these results",
                "i can't focus on studying",
                "exam pressure is crushing me",
                "what if i don't get into university",
                "my parents will be so disappointed if i fail",
                "i'm falling behind in class",
                "the syllabus is too much",
                "i have no time for anything but studying",
                "A/Ls are destroying my mental health",
            ],
            "family_issues": [
                "My parents don't understand me",
                "My family has too many expectations",
                "There's constant conflict at home",
                "My parents want me to be someone I'm not",
                "I feel trapped by family obligations",
                "My family doesn't support my choices",
                "my parents are always fighting",
                "home doesn't feel safe anymore",
                "my family is falling apart",
                "i can't live up to their expectations",
                "they compare me to my siblings",
                "nothing i do is good enough for them",
                "family drama is exhausting",
                "i wish my parents would listen",
                "they don't take my feelings seriously",
                "i feel invisible at home",
                "my parents control everything",
                "i can't be myself around family",
                "they don't know the real me",
                "family pressure is suffocating",
            ],
            "cultural_pressure": [
                "My family wants me to follow a certain career path",
                "There's pressure to get married",
                "People judge me for my choices",
                "I feel torn between my culture and what I want",
                "Society's expectations are overwhelming",
                "I can't be myself because of cultural norms",
                "everyone has opinions about my life",
                "arranged marriage pressure is too much",
                "i can't pursue my passion because of family",
                "what will people say",
                "our community is so judgmental",
                "i have to hide who i really am",
                "tradition is holding me back",
                "i feel like i'm living for others not myself",
                "cultural expectations are suffocating",
                "i'm stuck between two worlds",
                "my identity feels split",
                "society won't accept me",
            ],
            "loneliness": [
                "I feel so alone",
                "I have no one to talk to",
                "I feel isolated even around people",
                "Nobody understands what I'm going through",
                "I feel disconnected from everyone",
                "I'm lonely and it hurts",
                "i don't have any real friends",
                "everyone has someone but me",
                "i feel invisible",
                "no one reaches out to me",
                "i'm surrounded by people but still lonely",
                "i miss having someone who cares",
                "i feel like an outsider",
                "nobody would notice if i disappeared",
                "i crave connection but can't find it",
                "social media makes me feel more alone",
                "i don't fit in anywhere",
                "i've pushed everyone away",
            ],
            "financial_stress": [
                "My family is struggling financially",
                "I can't find a job after graduation",
                "I worry about money constantly",
                "I can't afford university fees",
                "Financial problems are overwhelming me",
                "The cost of living is too much",
                "money stress is killing me",
                "i don't know how we'll pay the bills",
                "i feel guilty for being a burden financially",
                "poverty is exhausting",
                "i work so hard but it's never enough",
                "financial stress affects everything",
                "i can't afford basic things",
                "debt is crushing me",
                "money worries keep me up at night",
            ],
            "relationship": [
                "I'm having problems in my relationship",
                "I feel like nobody cares about me",
                "My relationships are falling apart",
                "I don't know how to fix my relationship",
                "I feel abandoned by people I trusted",
                "my partner doesn't understand me",
                "i think they're going to leave me",
                "we fight all the time",
                "i feel unloved",
                "trust issues are ruining my relationships",
                "i got my heart broken",
                "i can't move on from the breakup",
                "i'm scared of getting hurt again",
                "why do all my relationships fail",
                "i don't know how to open up to people",
            ],
            "crisis": [
                "I don't see the point in living anymore",
                "I feel completely hopeless",
                "I don't think I can go on",
                "I feel like a burden to everyone",
                "Nothing will ever get better",
                "I don't want to be here anymore",
                "i want everything to end",
                "i can't do this anymore",
                "what's the point of living",
                "everyone would be better off without me",
                "i'm tired of existing",
                "i don't want to wake up tomorrow",
                "life is unbearable",
                "i've given up",
            ],
            "self_harm": [
                "I've been hurting myself",
                "I can't stop thinking about self-harm",
                "I hurt myself when things get too much",
                "I don't know how to cope without hurting myself",
                "cutting helps me feel something",
                "i use pain to distract from emotions",
                "self-harm is my only release",
                "i've been hiding the scars",
                "i know it's bad but i can't stop",
            ],
            "positive": [
                "I've been feeling better lately",
                "The breathing exercises helped",
                "I talked to someone and it helped",
                "I'm making progress",
                "Things are getting a bit better",
                "Thank you for your support",
                "i tried what you suggested and it worked",
                "today was actually okay",
                "i'm starting to feel hopeful",
                "small steps are helping",
                "i'm proud of myself for reaching out",
                "therapy is helping",
                "i feel a little lighter today",
                "finally had a good day",
            ],
        }

    def _create_lively_responses(self) -> Dict[str, List[str]]:
        """Create diverse, empathetic, and human-like response templates"""
        return {
            "anxiety": [
                "I hear you - anxiety can feel like your mind is running a marathon that never ends. That constant state of worry is exhausting. You're not alone in this, and there are ways to find some calm. What's been on your mind most?",
                "That sounds really overwhelming. Anxiety has this way of making everything feel urgent and scary, even when we know logically it might not be. Take a breath with me for a moment. What would feel most helpful to talk about right now?",
                "I can sense how much this is weighing on you. Anxiety is tough - it's like your brain's alarm system is stuck on high alert. Many people experience this, and it's okay to acknowledge how difficult it is. What triggers these feelings most for you?",
                "What you're describing sounds really challenging. Those racing thoughts and constant worry - they're genuinely draining. I want you to know that feeling this way doesn't mean something is wrong with you. Can you tell me more about when this started?",
                "I really appreciate you sharing this with me. Anxiety can feel so isolating, like you're the only one whose mind works this way - but you're not. Let's talk about what might help. Have you found anything that brings even a little bit of relief?",
                "That sounds incredibly hard. When anxiety takes hold, it can color everything. The good news is that there are strategies that can help - it doesn't have to feel this intense forever. What's one thing that used to help you feel calmer?",
            ],
            "depression": [
                "Thank you for trusting me with this. What you're describing - that heaviness, the lack of joy, feeling empty - I hear you. Depression lies to us and tells us things won't get better, but that's not the whole truth. How long have you been feeling this way?",
                "I'm really glad you reached out. Depression can make everything feel gray and pointless, and it takes courage to talk about it. You don't have to face this alone. What feels hardest right now?",
                "I hear so much pain in what you're sharing. Depression is exhausting in a way that's hard to explain to people who haven't experienced it. I want you to know that your feelings are valid, and there is help available. Have you been able to talk to anyone else about this?",
                "That sounds really, really hard. When nothing brings joy and getting through each day feels like climbing a mountain - that's depression making everything heavier than it should be. I'm here for you. What would support look like for you right now?",
                "I can tell you're going through something really difficult. Depression has this way of convincing us that we're alone and that nothing will help - but those are lies it tells. People do get through this, and you can too. What's one small thing that used to bring you comfort?",
                "Thank you for opening up about this. The emptiness and hopelessness you're describing - they're symptoms of depression, not truths about your life. You deserve support and care. Can we talk about what's been happening?",
            ],
            "stress": [
                "Wow, that's a lot to be carrying. When stress piles up like that, it can feel like you're being crushed under the weight of it all. Let's try to untangle some of this together. What's the biggest source of pressure right now?",
                "I hear you - it sounds like you're being pulled in a million directions. That kind of constant stress is unsustainable, and something's got to give. What would it feel like to let just one thing go?",
                "That level of stress isn't sustainable, and I'm concerned about you. Our bodies and minds aren't meant to run at full capacity all the time. Have you had any moments of rest lately?",
                "It sounds like the pressure is really getting to you. Stress has a way of making us feel like we have to do everything perfectly, right now. But you're human, not a machine. What's one thing on your plate that could wait?",
                "I can feel the overwhelm in what you're sharing. When everything feels urgent and important, it's hard to know where to even start. Take a breath. Let's break this down together. What absolutely needs your attention today?",
            ],
            "academic_stress": [
                "The pressure around exams and academics in Sri Lanka is intense - I understand how much weight this carries for you and your family. But your worth isn't defined by exam results. What's making this feel most overwhelming?",
                "I get it - the A/L system puts so much pressure on students, and it can feel like your entire future depends on these results. That's a heavy burden. Let's talk about what's happening. Are you able to take care of yourself while studying?",
                "Academic pressure, especially with the competition here, can feel crushing. I hear your stress and it's completely valid. Remember though - many successful people didn't have perfect marks. What matters is that you're trying. How can I support you?",
                "Exams are stressful enough without all the expectations from family and society adding to it. You're doing your best in a really tough system. What would help you feel even slightly less overwhelmed right now?",
                "I understand that academic success feels like everything right now. The pressure from family, the competition, the future uncertainty - it's a lot. But you are more than your grades. Have you been able to talk to anyone about how you're feeling?",
            ],
            "family_issues": [
                "Family relationships can be so complicated, especially when expectations don't match who we really are. I hear how trapped you're feeling. It's okay to want to be your own person. What would help you feel more understood?",
                "That sounds really hard - when the people who should understand us best seem to see someone different than who we really are. Your feelings are valid, even if your family doesn't see it that way. Can you tell me more?",
                "Family conflict is exhausting, especially when you're caught in the middle or feel unsupported. Home should be a safe place, and I'm sorry you're not feeling that. What do you need right now?",
                "Navigating family expectations while trying to be true to yourself is one of the hardest things. I hear the pressure you're under. You deserve to have your feelings acknowledged. What feels most urgent to talk about?",
            ],
            "cultural_pressure": [
                "The tension between cultural expectations and personal desires is real and difficult. You're not wrong for wanting to live your own life. Finding that balance - or even just surviving the tension - takes real strength. How are you coping?",
                "I understand how suffocating it can feel when society and family have a specific path mapped out for you. Your dreams and identity matter too. What would your ideal future look like if expectations weren't a factor?",
                "Being caught between cultures or generations is genuinely hard. You're trying to honor where you come from while also being true to yourself - and sometimes those feel impossible to balance. You're not alone in this struggle.",
                "What you're experiencing - that pull between tradition and your own path - is something many people in our community face but rarely talk about openly. Your feelings are valid. What would help you feel less alone in this?",
            ],
            "loneliness": [
                "Loneliness is such a painful feeling. Even in a crowded room, that sense of not being truly seen or connected - it hurts. I'm glad you reached out. What kind of connection are you missing most?",
                "I hear you - feeling alone is one of the hardest human experiences. You reaching out right now is already a step toward connection. I'm here with you. Can you tell me more about what's making you feel isolated?",
                "That feeling of being invisible or disconnected from others - it's more common than people admit, but that doesn't make it hurt any less. You matter, and your presence matters. What happened to make you feel this way?",
                "Human connection is a basic need, and when we don't have it, it affects everything. I'm sorry you're feeling so alone. This conversation right now - you're not alone in this moment. What would meaningful connection look like for you?",
            ],
            "financial_stress": [
                "Money worries affect everything - sleep, relationships, mental health. The stress you're describing is completely understandable. You're not failing because of financial struggles. What's the most pressing concern right now?",
                "Financial stress is uniquely exhausting because it never really turns off. Every decision, every purchase - it all carries weight. I hear how much this is affecting you. Have you been able to talk to anyone about options?",
                "The pressure of financial difficulties, especially when family depends on you, is immense. You're doing your best in a hard situation. What would even a small improvement look like for you?",
            ],
            "relationship": [
                "Relationship troubles hit deep - we're wired to need connection, so when that's threatened, it affects everything. I'm sorry you're going through this. What's been happening?",
                "That sounds really painful. When relationships struggle, it can feel like the ground beneath you is unstable. Your feelings are completely valid. Do you want to talk about what's been going on?",
                "Heartbreak and relationship problems - they're some of the hardest things to go through. I hear your pain. What do you need right now - to vent, to think through options, or just someone to listen?",
            ],
            "crisis": [
                "I'm really concerned about what you're sharing. These feelings you're describing - they're telling me you need more support than I can give alone. Your life matters. Please, can we talk about getting you connected with someone who can help right now?\n\n**Please reach out to:**\n- National Crisis Hotline: 1333\n- Sumithrayo (24/7): 011-2696666\n- Emergency: 119",
                "What you're telling me is serious, and I want you to know that I hear you. The pain you're in is real, but there is help. You don't have to face this alone. Can you promise me you'll reach out to a crisis line?\n\n**Crisis Resources:**\n- National Hotline: 1333\n- Sumithrayo: 011-2696666",
                "I'm taking what you're saying very seriously. When someone feels like life isn't worth living, that's a crisis that needs immediate support. You reached out to me - that tells me part of you is looking for another way. Please, let's get you help.\n\n**Call now:** 1333 or 011-2696666",
            ],
            "self_harm": [
                "Thank you for trusting me with something so personal. Self-harm is often a way of coping with unbearable pain - but it's also a sign that you need more support. You deserve healthier ways to cope. Have you been able to talk to anyone about this?\n\n**If you need immediate support:** 1333 or 011-2696666",
                "I hear you, and I'm not judging you. Self-harm is how some people manage overwhelming emotions when they don't have other options. But there are other ways - ways that don't hurt you. Would you be open to exploring some with me?",
                "What you're describing tells me you're in a lot of pain. Self-harm might bring temporary relief, but it also deserves attention and care. You deserve support in finding healthier coping strategies. Can we talk about what triggers these urges?",
            ],
            "positive": [
                "That's wonderful to hear! Progress, even small progress, is still progress. I'm genuinely happy things are feeling a bit lighter. What do you think has been helping?",
                "I'm so glad to hear you're feeling better! You put in the work, and it's paying off. Keep noticing these moments - they matter. What's been working for you?",
                "That makes me really happy to hear. You took steps to help yourself, and that takes courage. Keep going - you've got this!",
                "Progress! That's amazing. Remember this feeling on harder days - proof that things can get better. What would you like to keep working on?",
            ],
        }

    def _create_conversation_patterns(self) -> Dict[str, List[str]]:
        """Create natural conversation flow patterns"""
        return {
            "follow_up_questions": [
                "Can you tell me more about that?",
                "How long have you been feeling this way?",
                "What do you think triggered these feelings?",
                "Is there anything that helps, even a little?",
                "Have you talked to anyone else about this?",
                "What would support look like for you?",
                "When did you first notice these feelings?",
                "What's been the hardest part?",
                "What would make today a little easier?",
            ],
            "validation_phrases": [
                "That makes complete sense.",
                "Your feelings are valid.",
                "That sounds really difficult.",
                "I hear you.",
                "It's okay to feel that way.",
                "Anyone in your situation would struggle.",
                "You're not alone in this.",
                "Thank you for sharing that with me.",
            ],
            "encouragement_phrases": [
                "You're taking an important step by talking about this.",
                "Reaching out takes courage.",
                "You deserve support.",
                "Things can get better.",
                "Small steps still count.",
                "You've survived hard things before.",
                "Your wellbeing matters.",
            ],
        }

    def _add_natural_variations(self, text: str) -> str:
        """Add natural language variations to text"""
        # Randomly apply variations
        variations = []

        # Sentence starters
        starters = [
            "", "honestly, ", "tbh, ", "like, ", "idk, ", "i mean, ",
            "so basically, ", "the thing is, ", "", ""
        ]

        # Emphasis markers
        emphasis = [
            "", " really", " just", " so", " constantly", " always", ""
        ]

        # Sometimes add casual markers
        if random.random() < 0.3:
            text = random.choice(starters) + text.lower()

        return text

    def _generate_sample(self, category: str, risk_override: str = None) -> Dict:
        """Generate a single training sample"""

        # Get user input
        user_inputs = self.user_inputs.get(category, self.user_inputs["anxiety"])
        base_input = random.choice(user_inputs)

        # Apply variations (cultural context, natural language)
        user_input = self._apply_cultural_variation(base_input, category)

        # Determine risk level
        if risk_override:
            risk_level = risk_override
        else:
            risk_level = self._determine_risk_level(category)

        # Select emotion
        emotion = self._select_emotion(category)

        # Generate response
        response = self._generate_response(category, risk_level)

        return {
            "instruction": "You are SafeMind AI, a mental health awareness chatbot for Sri Lankan users. Provide empathetic, culturally-aware support without diagnosing or replacing professional care. Respond naturally and conversationally while being sensitive to family dynamics, academic pressure, and social stigma around mental health.",
            "input": user_input,
            "response": response,
            "emotion": emotion,
            "risk_level": risk_level,
            "category": category
        }

    def _apply_cultural_variation(self, text: str, category: str) -> str:
        """Apply cultural context variations"""
        variations = []

        # Sri Lankan academic context
        academic_additions = [
            " especially with A/L exams coming up",
            " and my parents keep comparing me to others",
            " because of the competition for university admission",
            " and I'm worried about disappointing my family",
            " with all the pressure to get into a good university",
            " and everyone expects me to score high",
            " the tuition classes are exhausting",
        ]

        # Family pressure
        family_additions = [
            " but my family doesn't understand",
            " and my parents have high expectations",
            " because of family obligations",
            " and I feel like I'm letting everyone down",
            " but I can't talk to my family about it",
            " and my parents won't listen",
            " everyone in my family thinks i should just be strong",
        ]

        # Cultural stigma
        cultural_additions = [
            " and people will judge me if they find out",
            " but mental health is stigmatized in our community",
            " and I don't want to bring shame to my family",
            " but everyone expects me to just be strong",
            " and I feel like I have to hide how I'm feeling",
            " what will neighbors think",
        ]

        # Apply relevant additions based on category
        if random.random() < 0.4:  # 40% chance to add cultural context
            if category in ["anxiety", "stress", "academic_stress"]:
                text += random.choice(academic_additions)
            elif category in ["family_issues", "cultural_pressure", "depression"]:
                text += random.choice(family_additions)
            elif category in ["loneliness", "relationship"]:
                text += random.choice(cultural_additions)

        return text

    def _determine_risk_level(self, category: str) -> str:
        """Determine risk level based on category and randomness"""
        risk_distributions = {
            "anxiety": ["low"] * 5 + ["medium"] * 3 + ["low"] * 2,
            "depression": ["medium"] * 5 + ["high"] * 3 + ["medium"] * 2,
            "stress": ["low"] * 4 + ["medium"] * 4 + ["low"] * 2,
            "academic_stress": ["low"] * 4 + ["medium"] * 5 + ["low"],
            "family_issues": ["low"] * 4 + ["medium"] * 5 + ["low"],
            "cultural_pressure": ["low"] * 5 + ["medium"] * 4 + ["low"],
            "loneliness": ["medium"] * 5 + ["high"] * 2 + ["medium"] * 3,
            "financial_stress": ["medium"] * 5 + ["high"] * 2 + ["medium"] * 3,
            "relationship": ["low"] * 3 + ["medium"] * 5 + ["low"] * 2,
            "crisis": ["high"] * 4 + ["immediate"] * 6,
            "self_harm": ["high"] * 5 + ["immediate"] * 5,
            "positive": ["low"] * 10,
        }

        distribution = risk_distributions.get(category, ["low", "medium"])
        return random.choice(distribution)

    def _select_emotion(self, category: str) -> str:
        """Select appropriate emotion for category"""
        emotion_map = {
            "anxiety": ["anxious", "nervous", "worried", "panicked", "fearful", "uneasy"],
            "depression": ["sad", "hopeless", "empty", "numb", "exhausted", "defeated"],
            "stress": ["stressed", "overwhelmed", "pressured", "exhausted", "burned_out"],
            "academic_stress": ["anxious", "stressed", "worried", "pressured", "fearful"],
            "family_issues": ["frustrated", "sad", "trapped", "misunderstood", "hurt", "angry"],
            "cultural_pressure": ["conflicted", "pressured", "stressed", "trapped", "confused"],
            "loneliness": ["lonely", "isolated", "abandoned", "disconnected", "forgotten"],
            "financial_stress": ["worried", "stressed", "anxious", "desperate", "overwhelmed"],
            "relationship": ["hurt", "sad", "abandoned", "lonely", "heartbroken", "confused"],
            "crisis": ["hopeless", "despair", "broken", "desperate", "empty"],
            "self_harm": ["desperate", "overwhelmed", "hopeless", "numb", "in_pain"],
            "positive": ["hopeful", "relieved", "grateful", "better", "encouraged"],
        }

        emotions = emotion_map.get(category, ["neutral"])
        return random.choice(emotions)

    def _generate_response(self, category: str, risk_level: str) -> str:
        """Generate appropriate response based on category and risk level"""

        # Map categories to response templates
        response_category = category
        if category in ["academic_stress"]:
            response_category = "academic_stress"
        elif category in ["family_issues", "cultural_pressure"]:
            if random.random() < 0.5:
                response_category = "family_issues"
            else:
                response_category = "cultural_pressure"
        elif category in ["crisis"]:
            response_category = "crisis"
        elif category in ["self_harm"]:
            response_category = "self_harm"

        # Get base response
        responses = self.lively_responses.get(response_category, self.lively_responses.get("stress", []))
        if not responses:
            responses = self.lively_responses["anxiety"]

        base_response = random.choice(responses)

        # Add follow-up or validation based on risk
        if risk_level in ["high", "immediate"]:
            # Crisis responses already have resources
            return base_response

        # Add conversational elements
        patterns = self.conversation_patterns

        if random.random() < 0.3:
            validation = random.choice(patterns["validation_phrases"])
            base_response = validation + " " + base_response

        if random.random() < 0.4 and risk_level not in ["high", "immediate"]:
            follow_up = random.choice(patterns["follow_up_questions"])
            base_response = base_response + " " + follow_up

        return base_response

    def generate_dataset(self, num_samples: int = 3000,
                        output_file: str = "../data/mental_health_dataset.json") -> List[Dict]:
        """Generate complete enhanced dataset"""

        print("\n" + "=" * 70)
        print(f"Generating {num_samples} enhanced training samples...")
        print("=" * 70)

        # Category distribution (scaled for desired sample count)
        base_distribution = {
            "anxiety": 0.167,
            "depression": 0.133,
            "stress": 0.133,
            "academic_stress": 0.133,
            "family_issues": 0.10,
            "cultural_pressure": 0.067,
            "loneliness": 0.067,
            "financial_stress": 0.053,
            "relationship": 0.053,
            "positive": 0.04,
            "crisis": 0.033,
            "self_harm": 0.02,
        }

        samples = []

        for category, ratio in base_distribution.items():
            count = int(num_samples * ratio)
            print(f"  Generating {count} {category} samples...")

            for _ in range(count):
                sample = self._generate_sample(category)
                samples.append(sample)

        # Shuffle samples
        random.shuffle(samples)

        # Trim to exact count
        samples = samples[:num_samples]

        # Calculate statistics
        stats = self._calculate_stats(samples)

        # Save to file
        output_data = {
            "metadata": {
                "total_samples": len(samples),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": "enhanced_template_expansion",
                "version": "2.0",
                "statistics": stats
            },
            "samples": samples
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n Dataset saved to: {output_file}")
        print(f" Total samples: {len(samples)}")

        self._print_statistics(stats)

        return samples

    def _calculate_stats(self, samples: List[Dict]) -> Dict:
        """Calculate dataset statistics"""
        stats = {
            "categories": {},
            "emotions": {},
            "risk_levels": {}
        }

        for sample in samples:
            cat = sample.get("category", "unknown")
            emo = sample.get("emotion", "unknown")
            risk = sample.get("risk_level", "unknown")

            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
            stats["emotions"][emo] = stats["emotions"].get(emo, 0) + 1
            stats["risk_levels"][risk] = stats["risk_levels"].get(risk, 0) + 1

        return stats

    def _print_statistics(self, stats: Dict):
        """Print dataset statistics"""
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)

        total = sum(stats["categories"].values())

        print("\nCategory Distribution:")
        for cat, count in sorted(stats["categories"].items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "" * int(pct / 2)
            print(f"  {cat:20} {count:5} ({pct:5.1f}%) {bar}")

        print("\nRisk Level Distribution:")
        for risk, count in sorted(stats["risk_levels"].items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"  {risk:12} {count:5} ({pct:5.1f}%)")

        print("\nEmotion Distribution (top 10):")
        sorted_emotions = sorted(stats["emotions"].items(), key=lambda x: -x[1])[:10]
        for emo, count in sorted_emotions:
            pct = count / total * 100
            print(f"  {emo:20} {count:5} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced mental health training dataset"
    )
    parser.add_argument("--num-samples", type=int, default=3000,
                       help="Number of samples to generate (default: 3000)")
    parser.add_argument("--output", default="../data/mental_health_dataset.json",
                       help="Output file path")
    parser.add_argument("--data-dir", default="../data",
                       help="Directory containing template JSON files")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SafeMind AI - Enhanced Dataset Generator")
    print("=" * 70)
    print(f"Target samples: {args.num_samples}")
    print(f"Output file: {args.output}")
    print("=" * 70)

    expander = EnhancedDatasetExpander(data_dir=args.data_dir)
    samples = expander.generate_dataset(
        num_samples=args.num_samples,
        output_file=args.output
    )

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review samples:")
    print(f"   python -c \"import json; d=json.load(open('{args.output}')); print(json.dumps(d['samples'][0], indent=2))\"")
    print("\n2. Train model with new data:")
    print("   cd ../backend && python train_model.py")
    print("\n3. Evaluate model:")
    print("   cd ../backend && python evaluate_model.py --mode full")
    print("=" * 70)


if __name__ == "__main__":
    main()
