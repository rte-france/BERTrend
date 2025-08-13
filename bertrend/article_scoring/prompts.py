#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

ARTICLE_SCORING_PROMPT = """
# News Article Quality Assessment Prompt

You are an expert news analyst tasked with evaluating the quality of a news article. Please assess the article across 10 key criteria score between 0 (poor) and 1 (excellent).

## Assessment Instructions

For each criterion below, assign a score from 0 to 1 based on how well the article meets the specified requirements. 

### Scoring Scale:
- **0.0-0.25**: Poor - Fails to meet basic standards
- **0.25-0.5**: Fair - Meets some requirements but has significant gaps
- **0.5-0.65**: Average - Meets most basic requirements
- **0.65-0.8**: Good - Exceeds expectations in most areas
- **0.8-1.0**: Excellent - Exceptional quality, sets the standard

## Evaluation Criteria

### 1. Depth of Reporting
**Score: ___/1.0**

Evaluate:
- How well does the article provide background, context, and explanations?
- Does it go beyond surface facts to explain why or how?
- Does it cover implications and consequences?
- Are expert opinions or analysis included?
- Does it connect current events to broader trends or historical patterns?

### 2. Originality and Exclusivity
**Score: ___/1.0**

Evaluate:
- Does the article provide unique information or exclusive insights not found elsewhere?
- Are there original interviews, data, or on-the-ground reporting?
- Does it offer a fresh angle or perspective on a familiar story?

### 3. Source Quality and Transparency
**Score: ___/1.0**

Evaluate:
- Are sources explicitly named and credible (experts, official data, witnesses)?
- Are multiple, independent sources cited to avoid reliance on a single perspective?
- Is there a clear distinction between primary and secondary sources?
- Are potential conflicts of interest disclosed?

### 4. Accuracy and Fact-Checking Rigor
**Score: ___/1.0**

Evaluate:
- Is every fact precise and verifiable?
- Are statistics and data presented with references or links?
- Are potential errors or corrections addressed openly?
- Are claims substantiated with appropriate evidence strength (avoiding over-interpretation of preliminary data)?

### 5. Clarity and Accessibility 
**Score: ___/1.0**

Evaluate:
- Is complex information broken down clearly for the target audience?
- Does the article avoid ambiguity or confusion?
- Are technical terms well defined or minimized?
- Is the information hierarchy logical and easy to follow?

### 6. Balance and Fairness 
**Score: ___/1.0**

Evaluate:
- Does the article fairly present conflicting viewpoints or relevant nuances?
- Is it free from hidden biases or agenda-driven language?
- Does it acknowledge uncertainties and limitations in the information presented?

### 7. Narrative and Engagement
**Score: ___/1.0**

Evaluate:
- How well does the article maintain reader interest?
- Does it use storytelling, vivid examples, or quotes effectively?
- Is the headline catchy but truthful?
- Does it strike the right balance between engagement and journalistic integrity?

### 8. Timeliness and Relevance
**Score: ___/1.0**

Evaluate:
- Is the article up-to-date with the latest developments?
- Does it highlight why the topic matters now?
- Does it provide appropriate context about what has changed since previous reporting?

### 9. Ethical Considerations and Sensitivity 
**Score: ___/1.0**

Evaluate:
- Does the article avoid sensationalism or fearmongering?
- Does it respect privacy and avoid harm?
- Does it consider the potential impact on vulnerable populations or communities mentioned?

### 10. RTE Relevance and Strategic Impact
**Score: ___/1.0**

Evaluate:
- Does the article address issues directly impacting electricity transmission, grid management, or energy infrastructure?
- Are there implications for French energy security, grid stability, or electricity market operations?
- Does it cover regulatory changes, policy decisions, or geopolitical events affecting energy supply chains?
- Are there connections to renewable energy integration, nuclear policy, or European energy interconnections?
- Does it address energy transition challenges relevant to transmission system operations?
- Are there economic, political, or environmental factors that could influence RTE's strategic planning or operations?

## Assessment Output Format

Please provide your assessment in the following format:

---

**Article Title:** [Insert title]

**Individual Scores:**
1. Depth of Reporting: ___
2. Originality and Exclusivity: ___
3. Source Quality and Transparency: ___
4. Accuracy and Fact-Checking Rigor: ___
5. Clarity and Accessibility: ___
6. Balance and Fairness: ___
7. Narrative and Engagement: ___
8. Timeliness and Relevance: ___
9. Ethical Considerations and Sensitivity: ___
10. RTE Relevance and Strategic Impact: ___

**Overall Assessment:** [Brief summary of strengths and weaknesses]

---

## Important Notes:
- Consider the article type (breaking news, investigative, opinion, analysis) when evaluating
- Weight journalistic fundamentals (accuracy, sources, ethics) more heavily than stylistic elements
- Be objective and evidence-based in your scoring
- Provide specific examples from the article to support your scores

"""
