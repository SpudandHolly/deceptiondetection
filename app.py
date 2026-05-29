"""
Deception Detection Web Application
Flask app that analyzes text for deception indicators
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from functools import wraps
import re
import os
import secrets
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Authentication - credentials from environment variables
AUTHORIZED_USERS = {
    os.environ.get('AUTH_EMAIL', 'jon@cavefish.co.uk'): os.environ.get('AUTH_PASSWORD', '')
}


def login_required(f):
    """Decorator to require login for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session['user'] not in AUTHORIZED_USERS:
            if request.is_json:
                return jsonify({'error': 'Unauthorized', 'login_required': True}), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@dataclass
class DeceptionIndicator:
    """A linguistic indicator of potential deception."""
    name: str
    description: str
    patterns: List[str]
    weight: float = 1.0
    category: str = "General"
    is_new: bool = False  # Flag for newly added indicators
    source: str = "public"  # "ddl" for DDL proprietary, "public" for established research
    examples: List[str] = field(default_factory=list)


class DeceptionDetector:
    """Deception detection model based on FSLA principles."""

    def __init__(self):
        self.indicators = self._initialize_indicators()

    def _initialize_indicators(self) -> List[DeceptionIndicator]:
        """Initialize all deception indicators."""
        return [
            # === PRONOUN ANALYSIS ===
            DeceptionIndicator(
                name="Pronoun Distancing",
                description="Avoiding 'I', 'my', 'we' when describing personal actions",
                patterns=[
                    r"\b(one|someone|people|they|you)\b.*\b(would|could|should|might)\b",
                    r"\bit was\b(?!.*\bI\b)",
                    r"\bthere was\b(?!.*\bI\b)",
                    r"\bthe\s+\w+\s+was\s+(done|made|taken|given)\b",
                ],
                weight=1.5,
                category="Pronouns"
            ),
            DeceptionIndicator(
                name="Pronoun Shift",
                description="Shifting from 'I' to 'we' or third person mid-statement",
                patterns=[
                    r"\bI\b.*\b(we|they|one)\b.*\b(did|went|made|took)\b",
                    r"\bwe\b.*\bbut\s+I\b",
                ],
                weight=1.3,
                category="Pronouns"
            ),

            # === TEMPORAL MARKERS ===
            DeceptionIndicator(
                name="Vague Temporal References",
                description="Imprecise time references suggesting uncertainty",
                patterns=[
                    r"\b(around|about|approximately|roughly|sometime)\s+(that|the|this)\s+(time|day|week|month|year)\b",
                    r"\bI\s+(think|believe|guess)\s+it\s+was\b",
                    r"\b(maybe|perhaps|possibly)\s+(around|about)\b",
                    r"\bat\s+some\s+point\b",
                ],
                weight=1.2,
                category="Temporal"
            ),
            DeceptionIndicator(
                name="Missing Time Periods",
                description="Gaps or jumps in chronological narrative",
                patterns=[
                    r"\b(then|next|after\s+that|later)\b.*\b(suddenly|just|immediately)\b",
                    r"\band\s+then\s+(suddenly|just)\b",
                    r"\bthe\s+next\s+thing\s+(I\s+)?(knew|remember)\b",
                ],
                weight=1.4,
                category="Temporal"
            ),

            # === LINGUISTIC HEDGING ===
            DeceptionIndicator(
                name="Excessive Hedging",
                description="Overuse of qualifying language that weakens commitment",
                patterns=[
                    r"\b(I\s+)?(think|believe|guess|suppose|assume|imagine)\s+(that|I|we|it)\b",
                    r"\b(probably|possibly|perhaps|maybe|might|could)\s+(be|have|had)\b",
                    r"\b(sort\s+of|kind\s+of|somewhat|relatively|fairly)\b",
                    r"\bto\s+(the\s+best\s+of\s+)?my\s+(knowledge|recollection|memory)\b",
                ],
                weight=1.3,
                category="Hedging"
            ),
            DeceptionIndicator(
                name="Qualifier Stacking",
                description="Multiple qualifiers indicating uncertainty or evasion",
                patterns=[
                    r"\b(probably|possibly|perhaps|maybe)\b.*\b(probably|possibly|perhaps|maybe)\b",
                    r"\b(I\s+think|believe)\b.*\b(probably|possibly|perhaps|maybe)\b",
                    r"\b(might|could)\s+have\s+(possibly|probably)\b",
                ],
                weight=1.5,
                category="Hedging"
            ),

            # === DENIAL ANALYSIS ===
            DeceptionIndicator(
                name="Weak Denial",
                description="Non-specific denials that don't directly address accusations",
                patterns=[
                    r"\bI\s+(would\s+)?(never|wouldn't)\s+(do\s+)?(something\s+like\s+)?that\b",
                    r"\bthat's\s+(not|just\s+not)\s+(true|right|correct)\b",
                    r"\bI\s+don't\s+know\s+(what|why|how)\s+you('re|'d)\b",
                    r"\bI\s+(have\s+)?(no\s+)?(idea|clue)\s+(what|why|how)\b",
                ],
                weight=1.4,
                category="Denials"
            ),
            DeceptionIndicator(
                name="Non-Denial Denial",
                description="Statements that appear to deny but don't address the allegation",
                patterns=[
                    r"\bI\s+(am\s+)?(not\s+)?(that\s+)?(kind|type|sort)\s+of\s+(person|man|woman)\b",
                    r"\b(people|those)\s+who\s+know\s+me\b",
                    r"\bI\s+have\s+(always|never)\s+(been|had)\b.*\b(integrity|honest|truthful)\b",
                    r"\byou\s+(can|should)\s+ask\s+anyone\b",
                ],
                weight=1.6,
                category="Denials"
            ),

            # === EMOTIONAL MARKERS ===
            DeceptionIndicator(
                name="Misplaced Emotion",
                description="Emotional responses inappropriate to topic or timing",
                patterns=[
                    r"\b(honestly|frankly|truthfully|to\s+be\s+honest)\b",
                    r"\bI\s+(swear|promise|guarantee)\b",
                    r"\b(believe\s+me|trust\s+me|you\s+have\s+to\s+understand)\b",
                    r"\bwhy\s+would\s+I\s+(lie|make\s+this\s+up)\b",
                ],
                weight=1.4,
                category="Emotional"
            ),
            DeceptionIndicator(
                name="Bolstering",
                description="Unnecessary emphasis on truthfulness or character",
                patterns=[
                    r"\bI'm\s+(an?\s+)?(honest|truthful|good)\s+(person|man|woman)\b",
                    r"\b(on\s+my|I\s+swear\s+on)\s+(mother's|father's|children's|life)\b",
                    r"\bas\s+God\s+(is\s+my\s+witness|as\s+my\s+witness)\b",
                    r"\bI\s+have\s+nothing\s+to\s+hide\b",
                ],
                weight=1.5,
                category="Emotional"
            ),

            # === DETAIL ANALYSIS ===
            DeceptionIndicator(
                name="Excessive Detail",
                description="Unnecessary specific details to appear credible",
                patterns=[
                    r"\bI\s+remember\s+(exactly|precisely|specifically|clearly)\b",
                    r"\bit\s+was\s+(exactly|precisely)\s+\d",
                    r"\b(the|this|that)\s+(exact|precise|specific)\s+(time|moment|day|place)\b",
                ],
                weight=1.2,
                category="Details"
            ),
            DeceptionIndicator(
                name="Lack of Detail",
                description="Missing sensory details or specific information",
                patterns=[
                    r"\b(something|stuff|things|whatever)\s+(happened|occurred|went\s+on)\b",
                    r"\bI\s+(don't|can't)\s+(remember|recall)\s+(the\s+)?(details|specifics)\b",
                    r"\bit\s+(all|just)\s+(happened|went)\s+(so\s+)?(fast|quickly)\b",
                ],
                weight=1.3,
                category="Details"
            ),

            # === COMMITMENT MARKERS ===
            DeceptionIndicator(
                name="Conditional Language",
                description="Using conditional tense when certainty expected",
                patterns=[
                    r"\bI\s+would\s+(say|think|believe|guess)\b",
                    r"\bif\s+I\s+(had\s+to|were\s+to)\s+(guess|say)\b",
                    r"\b(would|could|might)\s+have\s+(been|happened)\b",
                ],
                weight=1.2,
                category="Commitment"
            ),
            DeceptionIndicator(
                name="Passive Voice Evasion",
                description="Using passive voice to obscure responsibility",
                patterns=[
                    r"\b(was|were|been|being)\s+(done|made|taken|given|sent|told|asked|decided)\b",
                    r"\bit\s+(was|had\s+been)\s+(decided|determined|agreed)\b",
                    r"\b(mistakes|errors)\s+were\s+made\b",
                ],
                weight=1.3,
                category="Commitment"
            ),

            # === NARRATIVE STRUCTURE ===
            DeceptionIndicator(
                name="Out of Sequence",
                description="Events described out of chronological order",
                patterns=[
                    r"\b(oh|wait|actually|no)\s*,?\s*(I\s+)?(forgot|meant|should\s+have\s+said)\b",
                    r"\bgoing\s+back\s+to\b",
                    r"\bI\s+(should|need\s+to)\s+mention\b",
                ],
                weight=1.3,
                category="Narrative"
            ),
            DeceptionIndicator(
                name="Story Repair",
                description="Self-corrections or additions that change the narrative",
                patterns=[
                    r"\bwell\s*,?\s*(actually|no|wait)\b",
                    r"\bI\s+mean\b",
                    r"\bwhat\s+I\s+(meant|mean)\s+(was|is|to\s+say)\b",
                    r"\blet\s+me\s+(rephrase|clarify|explain)\b",
                ],
                weight=1.2,
                category="Narrative"
            ),

            # === SENSITIVITY INDICATORS ===
            DeceptionIndicator(
                name="Topic Avoidance",
                description="Changing subject or deflecting from question",
                patterns=[
                    r"\b(anyway|regardless|moving\s+on|but\s+the\s+point\s+is)\b",
                    r"\bthat's\s+not\s+(the|really\s+the)\s+(issue|point|question)\b",
                    r"\bwhat('s|'s)\s+(really|more)\s+important\s+(here\s+)?is\b",
                ],
                weight=1.4,
                category="Sensitivity"
            ),
            DeceptionIndicator(
                name="Memory Distancing",
                description="Creating psychological distance from recall",
                patterns=[
                    r"\bif\s+(I\s+)?(remember|recall)\s+(correctly|right|properly)\b",
                    r"\b(from\s+)?what\s+I\s+(can\s+)?(remember|recall)\b",
                    r"\bas\s+far\s+as\s+I\s+(can\s+)?(remember|recall|know)\b",
                    r"\bI\s+(don't|can't)\s+(seem\s+to\s+)?(remember|recall)\b",
                ],
                weight=1.2,
                category="Sensitivity"
            ),

            # === CORPORATE/FORMAL DECEPTION (DDL Proprietary) ===
            DeceptionIndicator(
                name="Corporate Double-Speak",
                description="Vague corporate language that obscures meaning",
                patterns=[
                    r"\b(going\s+forward|at\s+this\s+time|at\s+this\s+point\s+in\s+time)\b",
                    r"\b(synergies|leverage|optimize|streamline|rightsizing)\b",
                    r"\b(exploring\s+options|evaluating\s+alternatives|considering\s+possibilities)\b",
                    r"\b(challenges|headwinds|opportunities)\s+(ahead|remain)\b",
                ],
                weight=1.3,
                category="Corporate",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Earnings Call Red Flags",
                description="Patterns in misleading corporate communications",
                patterns=[
                    r"\b(we're|we\s+are)\s+(pleased|delighted|excited)\s+(to|with)\b",
                    r"\b(strong|robust|solid)\s+(performance|results|growth)\b.*\b(despite|notwithstanding)\b",
                    r"\b(one-time|non-recurring|exceptional)\s+(items|charges|events)\b",
                    r"\b(underlying|core|adjusted)\s+(performance|results|earnings)\b",
                ],
                weight=1.4,
                category="Corporate",
                source="ddl"
            ),

            # === MANIPULATION (DDL Proprietary) ===
            DeceptionIndicator(
                name="Love Bombing",
                description="Intense positive language used in manipulation",
                patterns=[
                    r"\b(you're|you\s+are)\s+(so|the\s+most)\s+(special|amazing|incredible|perfect)\b",
                    r"\b(never|no\s+one\s+has\s+ever)\s+(felt|met|known)\s+(anyone|someone)\s+like\b",
                    r"\b(destiny|fate|meant\s+to\s+be|soulmate)\b",
                    r"\b(only\s+you|you're\s+the\s+only\s+one)\b",
                ],
                weight=1.6,
                category="Manipulation",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Urgency Creation",
                description="Creating artificial time pressure",
                patterns=[
                    r"\b(right\s+now|immediately|urgent|asap|time\s+sensitive)\b",
                    r"\b(limited\s+time|act\s+now|don't\s+wait|hurry)\b",
                    r"\b(this\s+)?(opportunity|offer)\s+(won't|will\s+not)\s+(last|wait)\b",
                    r"\b(before\s+it's\s+too\s+late|last\s+chance)\b",
                ],
                weight=1.5,
                category="Manipulation",
                source="ddl"
            ),

            # === INSURANCE FRAUD (DDL Proprietary) ===
            DeceptionIndicator(
                name="Overly Specific Alibi",
                description="Rehearsed-sounding alibi with unnecessary details",
                patterns=[
                    r"\b(for\s+the\s+record|if\s+you\s+must\s+know|just\s+so\s+you\s+know)\b",
                    r"\b(I\s+was\s+at|I\s+had\s+been\s+at)\b.*\b(quiz|meeting|dinner|event|party|game)\b",
                    r"\b(I\s+came|I\s+finished|I\s+placed|I\s+won)\s+(first|second|third|last|\d+(?:st|nd|rd|th))\b",
                    r"\b(can\s+verify|will\s+confirm|can\s+vouch|witnesses?\s+who)\b",
                ],
                weight=1.6,
                category="Insurance",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Pre-emptive Explanation",
                description="Explaining away evidence gaps before being asked",
                patterns=[
                    r"\b(appear(s|ed)?\s+to\s+(have\s+)?be(en)?|must\s+have\s+been|clearly\s+were)\s+(professional|expert|experienced)\b",
                    r"\b(remarkably|surprisingly|unusually|strangely)\s+(little|few|no|minimal)\s+(sign|evidence|trace|damage|disturbance)\b",
                    r"\b(professional|expert|experienced)\s+(burglar|thie(f|ves)|criminal|job)\b",
                    r"\b(unfortunately|sadly|regrettably)\s+(I|we)\s+(don't|didn't|do\s+not)\s+have\b",
                    r"\b(no\s+longer\s+have|lost|misplaced|can't\s+find)\s+(the\s+)?(receipt|proof|documentation|evidence)\b",
                ],
                weight=1.7,
                category="Insurance",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Claim Urgency",
                description="Pushing for quick claim resolution",
                patterns=[
                    r"\b(as\s+)?(quickly|soon|fast)\s+as\s+possible\b",
                    r"\b(need|require|must\s+have)\s+(this|it)\s+(resolved|settled|processed|paid)\b",
                    r"\b(urgent|immediate|pressing)\s+(need|financial|circumstances)\b",
                    r"\b(hoping|expect|need)\s+(for\s+)?(a\s+)?(quick|swift|prompt|speedy)\s+(resolution|settlement|payment)\b",
                ],
                weight=1.5,
                category="Insurance",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Emotional Leverage",
                description="Emotional appeals in formal claims",
                patterns=[
                    r"\b(wife|husband|spouse|partner|child|mother|father)\s+(is|has\s+been|was)\s+(very\s+)?(distressed|upset|devastated|traumatised|traumatized|anxious|worried)\b",
                    r"\b(family|children|kids)\s+(are|have\s+been)\s+(very\s+)?(affected|impacted|struggling|suffering)\b",
                    r"\b(difficult|hard|tough|challenging)\s+(time|period|situation)\s+(for\s+(us|our\s+family|me))\b",
                    r"\b(sleepless\s+nights|can't\s+sleep|losing\s+sleep|anxiety)\b",
                ],
                weight=1.4,
                category="Insurance",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Convenient Documentation",
                description="Documentation claims that support the claim or explain gaps",
                patterns=[
                    r"\b(still\s+in|original)\s+(box|packaging|wrapper)\b",
                    r"\b(brand\s+new|never\s+(used|opened|worn))\b",
                    r"\b(have|can\s+provide)\s+(photographs|photos|pictures|documentation|receipts)\s+(of\s+)?(most|some)\b",
                    r"\b(upon|on)\s+request\b",
                    r"\b(provenance|authenticity)\s+(documented|certified|verified|available)\b",
                ],
                weight=1.3,
                category="Insurance",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Value Inflation Signals",
                description="Language associated with inflating claim values",
                patterns=[
                    r"\b(purchased\s+in|bought\s+in|acquired\s+in)\s+(geneva|switzerland|london|paris|dubai|hong\s+kong|new\s+york)\b",
                    r"\b(antique|vintage|rare|collectible|limited\s+edition|one\s+of\s+a\s+kind)\b",
                    r"\b(grandmother'?s?|grandfather'?s?|family\s+heirloom|passed\s+down|inherited)\b",
                    r"\b(irreplaceable|priceless|sentimental)\s+(value|worth|item)\b",
                    r"\b(top\s+of\s+the\s+(line|range)|premium|deluxe|luxury)\b",
                ],
                weight=1.4,
                category="Insurance",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Vague Item Descriptions",
                description="Non-specific descriptions making verification difficult",
                patterns=[
                    r"\b(various|assorted|miscellaneous|sundry)\s+(items|jewellery|jewelry|electronics|goods)\b",
                    r"\b(etc|et\s+cetera|and\s+so\s+on|and\s+more|among\s+others)\b",
                    r"\b(collection|selection)\s+of\b",
                    r"\b(other|additional|further)\s+(items|valuables|belongings)\b",
                ],
                weight=1.2,
                category="Insurance",
                source="ddl"
            ),
            DeceptionIndicator(
                name="Staged Discovery",
                description="Theatrical descriptions of discovering the incident",
                patterns=[
                    r"\b(to\s+my\s+horror|to\s+my\s+shock|to\s+my\s+surprise|to\s+my\s+dismay)\b",
                    r"\b(discovered|found|noticed)\s+that\s+.{0,30}\s+(had\s+been|was|were)\s+(entered|broken\s+into|burgled|ransacked)\b",
                    r"\b(couldn't\s+believe|could\s+not\s+believe|couldn't\s+believe\s+my\s+eyes)\b",
                ],
                weight=1.3,
                category="Insurance",
                source="ddl"
            ),

            # ============================================================
            # NEW INDICATORS - CBCA (Criteria-Based Content Analysis)
            # ============================================================
            DeceptionIndicator(
                name="Logical Inconsistencies",
                description="Contradictions or logical impossibilities in the narrative",
                patterns=[
                    r"\b(but\s+then\s+again|on\s+the\s+other\s+hand|actually\s+no|wait\s+no)\b",
                    r"\b(before|after)\b.{0,50}\b(before|after)\b",
                    r"\b(never|always)\b.{0,30}\b(except|but|although)\b",
                    r"\b(impossible|couldn't\s+have)\b.{0,30}\b(but|yet|still)\b",
                ],
                weight=1.7,
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="Overly Linear Narrative",
                description="Too structured/chronological - truthful accounts often jump around",
                patterns=[
                    r"\b(first|firstly)\b.{0,100}\b(second|secondly)\b.{0,100}\b(third|thirdly|finally)\b",
                    r"\b(step\s+one|step\s+1)\b.{0,100}\b(step\s+two|step\s+2)\b",
                    r"\b(then|next|after\s+that|subsequently)\b.{0,50}\b(then|next|after\s+that|subsequently)\b.{0,50}\b(then|next|after\s+that|subsequently)\b",
                ],
                weight=1.3,
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="Lack of Detail Quantity",
                description="Sparse on specifics where detail would be expected",
                patterns=[
                    r"\b(something|stuff|things)\s+(happened|occurred)\b",
                    r"\b(some|a\s+few|several)\s+(people|things|items)\b(?!.*\bspecifically\b)",
                    r"\b(somewhere|someplace|around\s+there)\b",
                    r"\b(a\s+while|some\s+time|a\s+bit)\s+(ago|later|after)\b",
                ],
                weight=1.4,
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="Missing Contextual Embedding",
                description="Lacks anchoring to time, place, or circumstances",
                patterns=[
                    r"^(?!.*\b(on|at|in|during)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(st|nd|rd|th)?|january|february|march|april|may|june|july|august|september|october|november|december)\b).{100,}$",
                ],
                weight=1.2,
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="No Direct Quotes",
                description="Absence of reproduced conversations - uses indirect speech only",
                patterns=[
                    r"\b(he|she|they)\s+(said|told|asked|mentioned)\s+(that|me|us)\b(?!.*[\"'])",
                    r"\b(we|they)\s+(talked|spoke|discussed)\s+(about)\b(?!.*[\"'])",
                    r"\b(according\s+to|apparently|supposedly)\b",
                ],
                weight=1.3,
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="Direct Quote Present",
                description="Reproduced conversation suggests genuine memory (positive indicator)",
                patterns=[
                    r"[\"'][A-Z][^\"']{10,}[\"']",
                    r"\b(he|she)\s+said\s*[,:]?\s*[\"']",
                    r"\bI\s+(said|asked|replied)\s*[,:]?\s*[\"']",
                ],
                weight=-0.5,  # Negative weight - reduces deception score
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="No Unexpected Complications",
                description="Story too smooth - real events have interruptions/obstacles",
                patterns=[
                    r"^(?!.*\b(unfortunately|but\s+then|however|suddenly|unexpectedly|to\s+my\s+surprise)\b).{200,}$",
                ],
                weight=1.2,
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="Spontaneous Corrections",
                description="Self-corrections suggest genuine recall (positive indicator)",
                patterns=[
                    r"\b(no\s+wait|actually|I\s+mean|sorry|correction|let\s+me\s+rephrase)\b",
                    r"\b(or\s+was\s+it|no\s+it\s+was|I\s+think\s+it\s+was\s+actually)\b",
                ],
                weight=-0.4,  # Negative weight - reduces deception score
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="Admits Memory Gaps",
                description="Honestly admitting not remembering (positive indicator)",
                patterns=[
                    r"\bI\s+(don't|can't|cannot)\s+(quite\s+)?(remember|recall)\s+(exactly|precisely|specifically)\b",
                    r"\b(my\s+memory\s+is|details\s+are)\s+(fuzzy|hazy|unclear|vague)\b",
                    r"\bI'm\s+not\s+(entirely\s+)?(sure|certain)\s+(about|of)\b",
                ],
                weight=-0.5,  # Negative weight - reduces deception score
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="Self-Deprecation Present",
                description="Admitting unfavorable details about self (positive indicator)",
                patterns=[
                    r"\bI\s+(should\s+have|shouldn't\s+have|could\s+have|made\s+a\s+mistake)\b",
                    r"\b(my\s+fault|I\s+was\s+wrong|I\s+admit)\b",
                    r"\b(stupid|foolish|naive|careless)\s+of\s+me\b",
                ],
                weight=-0.4,  # Negative weight - reduces deception score
                category="CBCA",
                is_new=True
            ),
            DeceptionIndicator(
                name="Unusual Specific Details",
                description="Unique specifics hard to invent (positive indicator)",
                patterns=[
                    r"\b(peculiar|odd|strange|unusual|distinctive)\s+(thing|detail|feature)\b",
                    r"\b(I\s+noticed|I\s+remember)\s+(specifically|particularly|distinctly)\b",
                    r"\b(the\s+exact|precisely|specifically)\s+(color|colour|shape|size|time|words)\b",
                ],
                weight=-0.3,  # Mild negative - can go either way
                category="CBCA",
                is_new=True
            ),

            # ============================================================
            # NEW INDICATORS - Reality Monitoring
            # ============================================================
            DeceptionIndicator(
                name="Lacks Sensory Details",
                description="Missing smell, taste, touch, sound - signs of fabrication",
                patterns=[
                    r"^(?!.*\b(smell|smelled|taste|tasted|felt|heard|sound|noise|touch|texture|cold|hot|warm|loud|quiet)\b).{150,}$",
                ],
                weight=1.4,
                category="Reality",
                is_new=True
            ),
            DeceptionIndicator(
                name="Rich Sensory Details",
                description="Contains perceptual information suggesting real memory (positive)",
                patterns=[
                    r"\b(smelled|smelt)\s+(like|of)\b",
                    r"\b(tasted|taste\s+of)\b",
                    r"\b(felt|feeling)\s+(cold|warm|hot|rough|smooth|soft|hard)\b",
                    r"\b(heard|sound|noise)\s+(of|like)\b",
                    r"\b(bright|dark|dim|loud|quiet|silent)\b",
                ],
                weight=-0.5,  # Negative weight - reduces deception score
                category="Reality",
                is_new=True
            ),
            DeceptionIndicator(
                name="Excessive Cognitive Operations",
                description="Too much 'I thought/realized/knew' - sign of construction",
                patterns=[
                    r"\bI\s+(thought|realized|knew|understood|figured|assumed)\s+(that|to\s+myself)?\b",
                    r"\bI\s+(was\s+thinking|kept\s+thinking|started\s+to\s+think)\b",
                    r"\b(it\s+occurred\s+to\s+me|I\s+came\s+to\s+realize)\b",
                    r"\bI\s+(suppose|presume|imagine|expect)\s+(that)?\b",
                ],
                weight=1.3,
                category="Reality",
                is_new=True
            ),
            DeceptionIndicator(
                name="Spatial Awareness Present",
                description="Clear sense of physical space/layout (positive indicator)",
                patterns=[
                    r"\b(to\s+the\s+left|to\s+the\s+right|in\s+front\s+of|behind|next\s+to|across\s+from)\b",
                    r"\b(about|approximately|roughly)\s+\d+\s*(feet|meters|metres|yards|inches)\b",
                    r"\b(facing|opposite|adjacent\s+to|near\s+the)\b",
                ],
                weight=-0.4,  # Negative weight - reduces deception score
                category="Reality",
                is_new=True
            ),
            DeceptionIndicator(
                name="Emotional State Described",
                description="Internal emotional experience suggests real memory (positive)",
                patterns=[
                    r"\bI\s+(felt|was\s+feeling)\s+(scared|afraid|terrified|anxious|nervous|relieved|happy|angry|confused|shocked)\b",
                    r"\b(my\s+heart|I\s+was\s+shaking|trembling|sweating|frozen)\b",
                    r"\b(panic|fear|relief|shock)\s+(set\s+in|washed\s+over|hit\s+me)\b",
                ],
                weight=-0.4,  # Negative weight - reduces deception score
                category="Reality",
                is_new=True
            ),

            # ============================================================
            # NEW INDICATORS - Statement Structure Analysis
            # ============================================================
            DeceptionIndicator(
                name="Imbalanced Statement Structure",
                description="Too much prologue, too little aftermath - classic deception pattern",
                patterns=[
                    r"^.{0,50}(before|prior\s+to|leading\s+up).{200,}(then|when).{0,100}$",
                ],
                weight=1.5,
                category="Structure",
                is_new=True
            ),
            DeceptionIndicator(
                name="Skipped Aftermath",
                description="Story ends abruptly after main event without follow-up",
                patterns=[
                    r"\b(and\s+that('s|\s+is)\s+(it|all|what\s+happened))\b",
                    r"\b(the\s+end|that's\s+basically\s+it|that's\s+my\s+story)\b",
                ],
                weight=1.3,
                category="Structure",
                is_new=True
            ),
            DeceptionIndicator(
                name="First Person Dropout",
                description="'I' disappears during critical parts of narrative",
                patterns=[
                    r"\b(the|a)\s+(door|window|car|item)\s+(was|were|had\s+been)\b(?!.*\bI\b.{0,30}$)",
                    r"\b(it|there)\s+(was|were)\s+(decided|agreed|done)\b",
                ],
                weight=1.5,
                category="Structure",
                is_new=True
            ),
            DeceptionIndicator(
                name="Left vs Went Usage",
                description="'Left' implies leaving something behind - often evasive",
                patterns=[
                    r"\bI\s+left\b(?!\s+(for|to\s+go|the))",
                    r"\b(we|they)\s+left\b(?!\s+(for|to\s+go|the))",
                ],
                weight=1.1,
                category="Structure",
                is_new=True
            ),
            DeceptionIndicator(
                name="Weak Opening Commitment",
                description="Fails to commit to core claim in opening",
                patterns=[
                    r"^(I\s+think|I\s+believe|As\s+far\s+as\s+I\s+know|To\s+be\s+honest)",
                    r"^(Basically|Essentially|In\s+essence|More\s+or\s+less)",
                ],
                weight=1.4,
                category="Structure",
                is_new=True
            ),

            # ============================================================
            # NEW INDICATORS - Linguistic Complexity / NLP
            # ============================================================
            DeceptionIndicator(
                name="Excessive Negations",
                description="Over-explaining what didn't happen rather than what did",
                patterns=[
                    r"\b(didn't|did\s+not|wasn't|were\s+not|hadn't|have\s+not|never)\b.*\b(didn't|did\s+not|wasn't|were\s+not|hadn't|have\s+not|never)\b",
                    r"\b(no|not|never|nothing|none|nobody)\b.{0,30}\b(no|not|never|nothing|none|nobody)\b",
                    r"\bI\s+(didn't|did\s+not)\s+\w+\s+(and|or)\s+(didn't|did\s+not)\b",
                ],
                weight=1.4,
                category="NLP",
                is_new=True
            ),
            DeceptionIndicator(
                name="Tense Inconsistency",
                description="Shifting between past and present tense inappropriately",
                patterns=[
                    r"\b(was|were|had)\b.{0,50}\b(is|are|has|have)\b.{0,50}\b(was|were|had)\b",
                    r"\b(said|told|went)\b.{0,30}\b(say|tell|go)\b",
                ],
                weight=1.3,
                category="NLP",
                is_new=True
            ),
            DeceptionIndicator(
                name="Causality Overuse",
                description="Excessive use of 'because/so that' - constructing justifications",
                patterns=[
                    r"\b(because|since|so\s+that|in\s+order\s+to|that's\s+why)\b.*\b(because|since|so\s+that|in\s+order\s+to|that's\s+why)\b",
                    r"\b(the\s+reason|this\s+is\s+why|that's\s+the\s+reason)\b",
                ],
                weight=1.2,
                category="NLP",
                is_new=True
            ),
            DeceptionIndicator(
                name="Certainty Undermining",
                description="Starts certain but undermines own statement",
                patterns=[
                    r"\b(definitely|certainly|absolutely)\b.{0,50}\b(I\s+think|maybe|probably|possibly)\b",
                    r"\b(I\s+know|I'm\s+sure)\b.{0,30}\b(but|although|however)\b",
                ],
                weight=1.4,
                category="NLP",
                is_new=True
            ),
            DeceptionIndicator(
                name="Distancing Temporal Language",
                description="Using past perfect to create psychological distance",
                patterns=[
                    r"\b(had\s+been|had\s+gone|had\s+done|had\s+said|had\s+made)\b",
                    r"\b(by\s+that\s+time|at\s+that\s+point|by\s+then)\b.{0,30}\b(had)\b",
                ],
                weight=1.1,
                category="NLP",
                is_new=True
            ),
            DeceptionIndicator(
                name="Filler Words Cluster",
                description="Clusters of filler words suggesting cognitive load",
                patterns=[
                    r"\b(um|uh|er|ah|like|you\s+know|basically|actually|literally)\b.{0,20}\b(um|uh|er|ah|like|you\s+know|basically|actually|literally)\b",
                    r"\b(kind\s+of|sort\s+of|I\s+guess|I\s+mean)\b.{0,20}\b(kind\s+of|sort\s+of|I\s+guess|I\s+mean)\b",
                ],
                weight=1.2,
                category="NLP",
                is_new=True
            ),
            DeceptionIndicator(
                name="Verb Phrase Simplification",
                description="Overly simple verb constructions under cognitive load",
                patterns=[
                    r"\b(I|he|she|we|they)\s+(went|got|did|made|took|put|said)\b.{0,20}\b(I|he|she|we|they)\s+(went|got|did|made|took|put|said)\b",
                ],
                weight=1.0,
                category="NLP",
                is_new=True
            ),
        ]

    def analyze(self, text: str) -> dict:
        """Analyze text and return results with highlighted matches."""
        indicators_found = []
        all_matches = []  # Store all match positions for highlighting
        total_weight = 0

        text_lower = text.lower()
        word_count = len(text.split())

        for indicator in self.indicators:
            matches = []
            match_positions = []

            for pattern in indicator.patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append(match.group())
                    match_positions.append({
                        'start': match.start(),
                        'end': match.end(),
                        'text': match.group(),
                        'indicator': indicator.name,
                        'category': indicator.category,
                        'weight': indicator.weight,
                        'is_new': indicator.is_new
                    })

            if matches:
                occurrence_rate = (len(matches) / max(word_count, 1)) * 100
                weighted_score = min(occurrence_rate * indicator.weight, 10)
                total_weight += weighted_score

                all_matches.extend(match_positions)

                indicators_found.append({
                    'name': indicator.name,
                    'description': indicator.description,
                    'category': indicator.category,
                    'matches': list(set(matches))[:5],
                    'count': len(matches),
                    'weight': indicator.weight,
                    'contribution': round(weighted_score, 2),
                    'is_new': indicator.is_new
                })

        # Calculate score
        base_score = min(total_weight * 2, 100)
        if word_count > 500:
            base_score *= 0.8
        elif word_count < 50:
            base_score *= 1.2

        score = round(min(max(base_score, 0), 100), 1)

        # Determine risk level
        if score < 20:
            risk_level = "Low"
            risk_color = "#22c55e"  # green
        elif score < 40:
            risk_level = "Medium"
            risk_color = "#eab308"  # yellow
        elif score < 60:
            risk_level = "High"
            risk_color = "#f97316"  # orange
        else:
            risk_level = "Very High"
            risk_color = "#ef4444"  # red

        # Generate highlighted HTML
        highlighted_html = self._generate_highlighted_html(text, all_matches)

        # Generate trend data (deception intensity across text)
        trend_data = self._generate_trend_data(text, all_matches)

        # Sort indicators by contribution
        indicators_found.sort(key=lambda x: x['contribution'], reverse=True)

        # Generate recommendations
        recommendations = self._generate_recommendations(indicators_found, risk_level)

        # Generate category breakdown for pie chart
        category_breakdown = self._generate_category_breakdown(indicators_found)

        # Generate red flags summary
        red_flags = self._generate_red_flags(indicators_found, all_matches)

        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(indicators_found)

        return {
            'score': score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'word_count': word_count,
            'indicators_count': len(indicators_found),
            'indicators': indicators_found,
            'highlighted_html': highlighted_html,
            'recommendations': recommendations,
            'trend_data': trend_data,
            'category_breakdown': category_breakdown,
            'red_flags': red_flags,
            'follow_up_questions': follow_up_questions
        }

    def _generate_highlighted_html(self, text: str, matches: list) -> str:
        """Generate HTML with highlighted suspicious text."""
        if not matches:
            return text.replace('\n', '<br>')

        # Sort matches by start position (reverse for replacement)
        matches.sort(key=lambda x: x['start'], reverse=True)

        # Remove overlapping matches (keep highest weight)
        filtered_matches = []
        for match in sorted(matches, key=lambda x: (-x['weight'], x['start'])):
            overlaps = False
            for existing in filtered_matches:
                if not (match['end'] <= existing['start'] or match['start'] >= existing['end']):
                    overlaps = True
                    break
            if not overlaps:
                filtered_matches.append(match)

        # Sort by position for replacement
        filtered_matches.sort(key=lambda x: x['start'], reverse=True)

        # Apply highlights
        result = text
        for match in filtered_matches:
            weight = match['weight']
            is_new = match.get('is_new', False)

            # Positive indicators (negative weight) - green
            if weight < 0:
                bg_color = "#bbf7d0"  # green-200
                border = "#22c55e"  # green-500
                text_color = "#166534"  # green-800
            # High weight deception indicators - red
            elif weight >= 1.5:
                bg_color = "#fecaca"  # red-200
                border = "#ef4444"  # red-500
                text_color = "#991b1b"  # red-800
            # Medium weight - orange
            elif weight >= 1.3:
                bg_color = "#fed7aa"  # orange-200
                border = "#f97316"  # orange-500
                text_color = "#9a3412"  # orange-800
            # Low weight - yellow
            else:
                bg_color = "#fef08a"  # yellow-200
                border = "#eab308"  # yellow-500
                text_color = "#854d0e"  # yellow-800

            # New indicators get cyan/teal tint overlay
            if is_new and weight > 0:
                bg_color = "#cffafe"  # cyan-100
                text_color = "#164e63"  # cyan-900
                border_style = f"2px dashed {border}"
            else:
                border_style = f"2px solid {border}"

            new_badge = ' [NEW]' if is_new else ''
            highlighted = f'<span class="highlight" style="background-color: {bg_color}; color: {text_color}; border-bottom: {border_style}; padding: 2px 4px; border-radius: 3px; cursor: help; font-weight: 500;" title="{match["indicator"]}{new_badge} ({match["category"]})">{match["text"]}</span>'
            result = result[:match['start']] + highlighted + result[match['end']:]

        return result.replace('\n', '<br>')

    def _generate_trend_data(self, text: str, matches: list, num_segments: int = 20) -> dict:
        """Generate trend data showing deception intensity across the text."""
        text_length = len(text)
        if text_length == 0 or not matches:
            return {
                'labels': [f'{i*5}%' for i in range(num_segments + 1)],
                'values': [0] * (num_segments + 1),
                'categories': {}
            }

        segment_size = text_length / num_segments
        segment_scores = [0.0] * (num_segments + 1)
        category_data = {}

        for match in matches:
            # Determine which segment(s) this match falls into
            start_segment = int(match['start'] / segment_size)
            end_segment = int(match['end'] / segment_size)

            # Clamp to valid range
            start_segment = min(start_segment, num_segments)
            end_segment = min(end_segment, num_segments)

            # Add weighted score to affected segments
            for seg in range(start_segment, end_segment + 1):
                segment_scores[seg] += match['weight']

            # Track by category
            cat = match['category']
            if cat not in category_data:
                category_data[cat] = [0.0] * (num_segments + 1)
            for seg in range(start_segment, end_segment + 1):
                category_data[cat][seg] += match['weight']

        # Normalize scores to 0-100 scale
        max_score = max(segment_scores) if max(segment_scores) > 0 else 1
        normalized_scores = [round((s / max_score) * 100, 1) for s in segment_scores]

        # Generate labels (position in text as percentage)
        labels = [f'{int(i * (100 / num_segments))}%' for i in range(num_segments + 1)]

        return {
            'labels': labels,
            'values': normalized_scores,
            'raw_values': [round(s, 2) for s in segment_scores],
            'categories': {cat: [round(v, 2) for v in vals] for cat, vals in category_data.items()}
        }

    def _generate_recommendations(self, indicators: list, risk_level: str) -> list:
        """Generate recommendations based on findings."""
        recommendations = []

        if risk_level == "Low":
            recommendations.append({
                'type': 'success',
                'text': 'No significant deception indicators detected. Standard verification applies.'
            })
            return recommendations

        indicator_names = {ind['name'] for ind in indicators}
        categories = {ind['category'] for ind in indicators}

        if "Insurance" in categories:
            insurance_count = sum(1 for ind in indicators if ind['category'] == 'Insurance')
            if insurance_count >= 3:
                recommendations.append({
                    'type': 'danger',
                    'text': 'MULTIPLE INSURANCE FRAUD INDICATORS: Recommend SIU referral and detailed investigation.'
                })

        if "Overly Specific Alibi" in indicator_names:
            recommendations.append({
                'type': 'warning',
                'text': 'Verify alibi independently. Overly detailed alibis often indicate rehearsal.'
            })

        if "Pre-emptive Explanation" in indicator_names:
            recommendations.append({
                'type': 'warning',
                'text': 'Subject is explaining away evidence gaps before being asked. Request police report.'
            })

        if "Weak Denial" in indicator_names or "Non-Denial Denial" in indicator_names:
            recommendations.append({
                'type': 'info',
                'text': 'Request a direct, specific denial: first person, past tense, addressing the specific allegation.'
            })

        if "Excessive Hedging" in indicator_names or "Qualifier Stacking" in indicator_names:
            recommendations.append({
                'type': 'info',
                'text': 'Press for definitive answers without qualifiers.'
            })

        if "Bolstering" in indicator_names or "Misplaced Emotion" in indicator_names:
            recommendations.append({
                'type': 'info',
                'text': 'Focus on facts, not character assertions. Emphasis on truthfulness indicates sensitivity.'
            })

        if "Value Inflation Signals" in indicator_names or "Convenient Documentation" in indicator_names:
            recommendations.append({
                'type': 'warning',
                'text': 'Verify claimed values with receipts, bank statements, or independent appraisal.'
            })

        if "Love Bombing" in indicator_names or "Urgency Creation" in indicator_names:
            recommendations.append({
                'type': 'danger',
                'text': 'MANIPULATION TACTICS DETECTED: Slow down and verify all claims independently.'
            })

        if "Corporate Double-Speak" in indicator_names:
            recommendations.append({
                'type': 'info',
                'text': 'Request specific metrics and compare to previous communications.'
            })

        if risk_level in ["High", "Very High"]:
            recommendations.append({
                'type': 'warning',
                'text': 'Consider independent verification of all key claims before proceeding.'
            })

        return recommendations

    def _generate_category_breakdown(self, indicators: list) -> dict:
        """Generate category breakdown for pie chart."""
        category_counts = {}
        category_weights = {}

        for ind in indicators:
            cat = ind['category']
            if cat not in category_counts:
                category_counts[cat] = 0
                category_weights[cat] = 0
            category_counts[cat] += 1
            category_weights[cat] += ind['contribution']

        # Sort by weight contribution
        sorted_cats = sorted(category_weights.items(), key=lambda x: x[1], reverse=True)

        # Category colors
        colors = {
            'Insurance': '#ef4444',
            'Pronouns': '#f97316',
            'Temporal': '#eab308',
            'Hedging': '#84cc16',
            'Denials': '#22c55e',
            'Emotional': '#14b8a6',
            'Details': '#06b6d4',
            'Commitment': '#3b82f6',
            'Narrative': '#8b5cf6',
            'Sensitivity': '#a855f7',
            'Corporate': '#ec4899',
            'Manipulation': '#f43f5e',
            'CBCA': '#06b6d4',
            'Reality': '#10b981',
            'Structure': '#6366f1',
            'NLP': '#8b5cf6'
        }

        return {
            'labels': [cat for cat, _ in sorted_cats],
            'values': [round(weight, 1) for _, weight in sorted_cats],
            'counts': [category_counts[cat] for cat, _ in sorted_cats],
            'colors': [colors.get(cat, '#666666') for cat, _ in sorted_cats]
        }

    def _generate_red_flags(self, indicators: list, matches: list) -> list:
        """Generate concise red flag bullet points."""
        red_flags = []

        # Get unique indicator names with high contribution
        high_contrib = [ind for ind in indicators if ind['contribution'] > 1.0]

        for ind in high_contrib[:8]:  # Top 8 red flags
            # Create contextual red flag description
            match_sample = ind['matches'][0] if ind['matches'] else ''

            if ind['category'] == 'Insurance':
                if 'Alibi' in ind['name']:
                    red_flags.append(f"Unusually specific alibi details provided unprompted: \"{match_sample[:50]}...\"")
                elif 'Pre-emptive' in ind['name']:
                    red_flags.append(f"Pre-emptively explains evidence gaps before being questioned")
                elif 'Urgency' in ind['name']:
                    red_flags.append(f"Pushes for rapid claim resolution")
                elif 'Value' in ind['name']:
                    red_flags.append(f"Value inflation signals detected (luxury locations, heirlooms)")
                elif 'Documentation' in ind['name']:
                    red_flags.append(f"Convenient documentation claims that are hard to verify")
                elif 'Emotional' in ind['name']:
                    red_flags.append(f"Uses emotional leverage (family distress) in formal claim")
                else:
                    red_flags.append(f"{ind['name']}: {ind['description']}")

            elif ind['category'] == 'Pronouns':
                if 'Distancing' in ind['name']:
                    red_flags.append(f"Subject distances self using third-person or passive voice during key events")
                elif 'Shift' in ind['name']:
                    red_flags.append(f"Pronoun shifts from 'I' to 'we/they' mid-narrative")
                else:
                    red_flags.append(f"{ind['name']}: found \"{match_sample[:40]}\"")

            elif ind['category'] == 'Hedging':
                red_flags.append(f"Excessive hedging/qualifiers undermine statement certainty")

            elif ind['category'] == 'Denials':
                red_flags.append(f"Non-specific denial that doesn't directly address allegation")

            elif ind['category'] == 'Manipulation':
                red_flags.append(f"Manipulation tactic detected: {ind['name'].lower()}")

            elif ind['category'] == 'CBCA':
                if 'Logical' in ind['name']:
                    red_flags.append(f"Logical inconsistencies or contradictions in narrative")
                elif 'Linear' in ind['name']:
                    red_flags.append(f"Overly structured narrative - authentic accounts typically jump around")
                else:
                    red_flags.append(f"CBCA concern: {ind['description']}")

            elif ind['category'] == 'Reality':
                if 'Sensory' in ind['name'] and ind['weight'] > 0:
                    red_flags.append(f"Lacks sensory details (smell, sound, touch) expected in genuine memory")
                elif 'Cognitive' in ind['name']:
                    red_flags.append(f"Excessive cognitive operations ('I thought', 'I realized') suggest construction")
                else:
                    red_flags.append(f"{ind['name']}: {ind['description']}")

            else:
                red_flags.append(f"{ind['name']}: \"{match_sample[:40]}{'...' if len(match_sample) > 40 else ''}\"")

        return red_flags

    def _generate_follow_up_questions(self, indicators: list) -> list:
        """Generate suggested follow-up questions based on indicators found."""
        questions = []
        indicator_names = {ind['name'] for ind in indicators}
        categories = {ind['category'] for ind in indicators}

        # Insurance-specific questions
        if 'Insurance' in categories:
            if 'Overly Specific Alibi' in indicator_names:
                questions.append({
                    'question': "Can you describe the evening in reverse order, starting from when you discovered the break-in?",
                    'rationale': "Fabricated alibis are harder to recall backwards"
                })
                questions.append({
                    'question': "Who else was at the quiz night? Can you provide contact details for verification?",
                    'rationale': "Tests willingness to have alibi independently verified"
                })

            if 'Pre-emptive Explanation' in indicator_names:
                questions.append({
                    'question': "How do you know the burglars were professionals?",
                    'rationale': "Challenges unsolicited characterization of perpetrators"
                })
                questions.append({
                    'question': "Can you provide the police incident number and attending officer's name?",
                    'rationale': "Verifies formal report was made"
                })

            if 'Value Inflation Signals' in indicator_names:
                questions.append({
                    'question': "Do you have the original purchase receipts, bank statements, or credit card records for these items?",
                    'rationale': "Tests ability to substantiate claimed values"
                })
                questions.append({
                    'question': "When was the last independent valuation of the antique items?",
                    'rationale': "Challenges unverified value claims"
                })

            if 'Claim Urgency' in indicator_names:
                questions.append({
                    'question': "Is there a specific reason you need this resolved quickly?",
                    'rationale': "Explores motivation behind urgency"
                })

        # Pronoun-related questions
        if 'Pronoun Distancing' in indicator_names or 'First Person Dropout' in indicator_names:
            questions.append({
                'question': "I notice you said 'the window was broken' - can you tell me exactly what YOU saw and did when you arrived home?",
                'rationale': "Re-centers narrative on subject's direct experience"
            })

        # Hedging-related questions
        if 'Excessive Hedging' in indicator_names or 'Qualifier Stacking' in indicator_names:
            questions.append({
                'question': "You said you 'think' or 'believe' several times - can you tell me what you KNOW for certain?",
                'rationale': "Pushes for definitive statements"
            })

        # Denial-related questions
        if 'Weak Denial' in indicator_names or 'Non-Denial Denial' in indicator_names:
            questions.append({
                'question': "I need a direct answer: Did you [specific allegation]? Yes or no.",
                'rationale': "Forces specific denial rather than character defense"
            })

        # Memory/detail questions
        if 'Lack of Detail' in indicator_names or 'Lacks Sensory Details' in indicator_names:
            questions.append({
                'question': "Close your eyes and picture the scene. What did you hear? What did you smell? How did you feel physically?",
                'rationale': "Genuine memories contain multi-sensory details"
            })

        # CBCA-related questions
        if 'CBCA' in categories:
            if 'No Direct Quotes' in indicator_names:
                questions.append({
                    'question': "Did anyone say anything to you? Can you remember their exact words?",
                    'rationale': "Reproduced dialogue is a truthfulness indicator"
                })

            if 'Overly Linear Narrative' in indicator_names:
                questions.append({
                    'question': "Is there anything you forgot to mention, or anything that happened out of sequence?",
                    'rationale': "Authentic accounts often have corrections and insertions"
                })

        # Manipulation questions
        if 'Manipulation' in categories:
            questions.append({
                'question': "I'd like to take some time to verify these details. Can we schedule a follow-up in a few days?",
                'rationale': "Removes artificial urgency"
            })

        return questions[:6]  # Return top 6 most relevant questions


# Initialize detector
detector = DeceptionDetector()

# Store for custom weights (in production, use database)
custom_weights = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare')
def compare():
    return render_template('compare.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        password = request.form.get('password', '')

        if email in AUTHORIZED_USERS and AUTHORIZED_USERS[email] == password:
            session['user'] = email
            next_url = request.args.get('next', url_for('rules'))
            return redirect(next_url)
        else:
            error = 'Invalid email or password.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/rules')
@login_required
def rules():
    return render_template('rules.html', user=session.get('user'))


@app.route('/api/rules', methods=['GET'])
@login_required
def get_rules():
    """Get all rules with current weights."""
    rules = []
    for i, ind in enumerate(detector.indicators):
        rules.append({
            'id': i,
            'name': ind.name,
            'description': ind.description,
            'category': ind.category,
            'patterns': ind.patterns,
            'weight': custom_weights.get(i, ind.weight),
            'default_weight': ind.weight,
            'enabled': custom_weights.get(f'{i}_enabled', True),
            'is_new': ind.is_new,
            'source': ind.source  # "ddl" or "public"
        })
    return jsonify(rules)


@app.route('/api/rules/<int:rule_id>', methods=['PUT'])
@login_required
def update_rule(rule_id):
    """Update a specific rule's weight or patterns."""
    data = request.get_json()

    if rule_id < 0 or rule_id >= len(detector.indicators):
        return jsonify({'error': 'Invalid rule ID'}), 400

    # Update weight
    if 'weight' in data:
        weight = float(data['weight'])
        if 0 <= weight <= 3:
            custom_weights[rule_id] = weight
            detector.indicators[rule_id].weight = weight

    # Update enabled status
    if 'enabled' in data:
        custom_weights[f'{rule_id}_enabled'] = data['enabled']

    # Update patterns
    if 'patterns' in data:
        patterns = data['patterns']
        if isinstance(patterns, list):
            detector.indicators[rule_id].patterns = patterns

    # Update description
    if 'description' in data:
        detector.indicators[rule_id].description = data['description']

    return jsonify({'success': True, 'rule_id': rule_id})


@app.route('/api/rules/reset', methods=['POST'])
@login_required
def reset_rules():
    """Reset all rules to defaults."""
    global custom_weights
    custom_weights = {}
    # Reinitialize detector
    detector.indicators = detector._initialize_indicators()
    return jsonify({'success': True})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text.strip():
            return jsonify({'error': 'Please enter some text to analyze'})

        result = detector.analyze(text)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
