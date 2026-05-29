"""
Deception Detection Web Application
Flask app that analyzes text for deception indicators
"""

from flask import Flask, render_template, request, jsonify
import re
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

app = Flask(__name__)

@dataclass
class DeceptionIndicator:
    """A linguistic indicator of potential deception."""
    name: str
    description: str
    patterns: List[str]
    weight: float = 1.0
    category: str = "General"
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

            # === CORPORATE/FORMAL DECEPTION ===
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
                category="Corporate"
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
                category="Corporate"
            ),

            # === MANIPULATION ===
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
                category="Manipulation"
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
                category="Manipulation"
            ),

            # === INSURANCE FRAUD ===
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
                category="Insurance"
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
                category="Insurance"
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
                category="Insurance"
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
                category="Insurance"
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
                category="Insurance"
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
                category="Insurance"
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
                category="Insurance"
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
                category="Insurance"
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
                        'weight': indicator.weight
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
                    'contribution': round(weighted_score, 2)
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

        # Sort indicators by contribution
        indicators_found.sort(key=lambda x: x['contribution'], reverse=True)

        # Generate recommendations
        recommendations = self._generate_recommendations(indicators_found, risk_level)

        return {
            'score': score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'word_count': word_count,
            'indicators_count': len(indicators_found),
            'indicators': indicators_found,
            'highlighted_html': highlighted_html,
            'recommendations': recommendations
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
            if weight >= 1.5:
                bg_color = "#fecaca"  # red-200
                border = "#ef4444"  # red-500
                text_color = "#991b1b"  # red-800
            elif weight >= 1.3:
                bg_color = "#fed7aa"  # orange-200
                border = "#f97316"  # orange-500
                text_color = "#9a3412"  # orange-800
            else:
                bg_color = "#fef08a"  # yellow-200
                border = "#eab308"  # yellow-500
                text_color = "#854d0e"  # yellow-800

            highlighted = f'<span class="highlight" style="background-color: {bg_color}; color: {text_color}; border-bottom: 2px solid {border}; padding: 2px 4px; border-radius: 3px; cursor: help; font-weight: 500;" title="{match["indicator"]} ({match["category"]})">{match["text"]}</span>'
            result = result[:match['start']] + highlighted + result[match['end']:]

        return result.replace('\n', '<br>')

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


# Initialize detector
detector = DeceptionDetector()

# Store for custom weights (in production, use database)
custom_weights = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/rules')
def rules():
    return render_template('rules.html')


@app.route('/api/rules', methods=['GET'])
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
            'enabled': custom_weights.get(f'{i}_enabled', True)
        })
    return jsonify(rules)


@app.route('/api/rules/<int:rule_id>', methods=['PUT'])
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
def reset_rules():
    """Reset all rules to defaults."""
    global custom_weights
    custom_weights = {}
    # Reinitialize detector
    detector.indicators = detector._initialize_indicators()
    return jsonify({'success': True})


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'Please enter some text to analyze'})

    result = detector.analyze(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
