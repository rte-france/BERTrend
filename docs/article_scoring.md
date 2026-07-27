# Article Scoring System

The BERTrend article scoring system provides comprehensive quality assessment for news articles and reports using a multi-criteria evaluation framework.

## Overview

The article scoring system evaluates articles across 10 key quality criteria, each with configurable weights, to produce a final quality score and level. This system is particularly useful for:

- **Content Quality Assessment**: Automated evaluation of article quality across multiple dimensions
- **Newsletter Curation**: Selecting high-quality articles for newsletter distribution
- **Content Strategy**: Understanding strengths and weaknesses in content sources
- **Quality Monitoring**: Tracking content quality trends over time

## Core Components

### 1. Quality Levels (`QualityLevel`)

Articles are classified into five quality levels based on their final score:

- **POOR** (0.0-0.25): Low-quality content requiring significant improvement
- **FAIR** (0.25-0.50): Below-average content with notable issues
- **AVERAGE** (0.50-0.65): Acceptable content meeting basic standards
- **GOOD** (0.65-0.8): High-quality content with minor areas for improvement
- **EXCELLENT** (0.8-1.0): Outstanding content meeting the highest standards

### 2. Evaluation Criteria (`CriteriaScores`)

Each article is evaluated across 10 comprehensive criteria:

#### Content Quality Criteria

1. **Depth of Reporting** (Weight: 15%)
   - Background information and context provision
   - Connection to broader trends and implications
   - Thoroughness of analysis and explanation

2. **Originality and Exclusivity** (Weight: 10%)
   - Unique information and insights
   - Exclusive reporting and fresh perspectives
   - Novel angles on familiar topics

3. **Source Quality and Transparency** (Weight: 15%)
   - Quality and credibility of sources
   - Transparency in source attribution
   - Independence and conflict disclosure

4. **Accuracy and Fact-Checking Rigor** (Weight: 15%)
   - Precision and verifiability of information
   - Appropriate evidence strength
   - Error handling and corrections

#### Communication and Presentation

5. **Clarity and Accessibility** (Weight: 10%)
   - Clear communication and logical structure
   - Accessibility for target audience
   - Effective use of language and terminology

6. **Balance and Fairness** (Weight: 10%)
   - Fair presentation of multiple viewpoints
   - Absence of obvious bias
   - Acknowledgment of uncertainties and limitations

7. **Narrative and Engagement** (Weight: 10%)
   - Reader engagement and storytelling effectiveness
   - Balance between engagement and journalistic integrity
   - Compelling presentation without sensationalism

#### Timeliness and Relevance

8. **Timeliness and Relevance** (Weight: 5%)
   - Currency of information
   - Relevance to current context
   - Appropriate timing of publication

9. **Ethical Considerations and Sensitivity** (Weight: 5%)
   - Adherence to ethical reporting standards
   - Sensitivity to affected populations
   - Avoidance of harmful sensationalism

#### Domain-Specific Relevance

10. **RTE Relevance and Strategic Impact** (Weight: 5%)
    - Relevance to RTE operations and interests
    - Impact on energy transmission and grid management
    - Strategic planning implications

### 3. Weight Configuration (`WeightConfig`)

The scoring system uses configurable weights for each criterion, allowing customization based on specific needs:

```python
# Default weights
weights = WeightConfig()

# Custom weights
custom_weights = WeightConfig(
    depth_of_reporting=0.20,
    source_quality_and_transparency=0.20,
    accuracy_and_fact_checking_rigor=0.20,
)
```

**Weight Validation Rules:**
- Each weight must be between 0.0 and 1.0
- Total weights cannot exceed the number of criteria (10.0)
- Weights are normalized during final score calculation

### 4. Article Scoring (`ArticleScore`)

The main scoring class that combines individual criteria scores with weights to produce comprehensive assessment:

```python
from bertrend.article_scoring.article_scoring import (
    CriteriaScores,
    WeightConfig,
    ArticleScore,
)

# Create criteria scores
scores = CriteriaScores(
    depth_of_reporting=0.8,
    originality_and_exclusivity=0.7,
    source_quality_and_transparency=0.9,
    accuracy_and_fact_checking_rigor=0.85,
    clarity_and_accessibility=0.75,
    balance_and_fairness=0.8,
    narrative_and_engagement=0.7,
    timeliness_and_relevance=0.9,
    ethical_considerations_and_sensitivity=0.95,
    rte_relevance_and_strategic_impact=0.6,
)

# Create article score with default weights
article_score = ArticleScore(
    scores=scores, assessment_summary="Well-researched article with strong sources"
)

# Get final score and quality level
final_score = article_score.final_score
quality_level = article_score.quality_level
```

## Key Features

### Detailed Breakdown Analysis

Get comprehensive score breakdowns showing individual contributions:

```python
breakdown = article_score.get_detailed_breakdown()
# Returns: {
#     "depth_of_reporting": {
#         "score": 0.8,
#         "weight": 0.15,
#         "weighted_contribution": 0.120
#     },
#     ...
# }
```

### Strengths and Weaknesses Identification

Identify top performing and underperforming criteria:

```python
# Get top 3 strengths
strengths = article_score.get_top_strengths(n=3)
# Returns: [("ethical_considerations_and_sensitivity", 0.95), ...]

# Get top 3 weaknesses
weaknesses = article_score.get_top_weaknesses(n=3)
# Returns: [("rte_relevance_and_strategic_impact", 0.6), ...]
```

### Comprehensive Reporting

Generate detailed assessment reports:

```python
# Generate full report with breakdown
report = article_score.to_report(include_breakdown=True)
print(report)

# Export to dictionary for further processing
data = article_score.export_to_dict()
```

## Scoring Algorithm

The final score is calculated using the following formula:

```
final_score = (Σ(criterion_score × criterion_weight) / number_of_criteria)
```

This approach:
1. **Weights individual criteria** based on their importance
2. **Normalizes the result** by dividing by the number of criteria
3. **Ensures scores remain** between 0.0 and 1.0
4. **Allows flexible weighting** while maintaining consistency

## Usage Examples

### Basic Article Assessment

```python
# Simple assessment with default weights
scores = CriteriaScores(
    depth_of_reporting=0.7,
    originality_and_exclusivity=0.6,
    # ... other criteria
)

article_score = ArticleScore(scores=scores)
print(f"Quality: {article_score.quality_level}")
print(f"Score: {article_score.final_score}")
```

### Custom Weight Configuration

```python
# Emphasize accuracy and sources for news content
news_weights = WeightConfig(
    accuracy_and_fact_checking_rigor=0.25,
    source_quality_and_transparency=0.25,
    depth_of_reporting=0.20,
)

article_score = ArticleScore(scores=scores, weights=news_weights)
```

### Batch Processing

```python
def assess_articles(articles_data):
    results = []
    for article_data in articles_data:
        scores = CriteriaScores(**article_data["criteria_scores"])
        article_score = ArticleScore(scores=scores)
        results.append(
            {
                "id": article_data["id"],
                "final_score": article_score.final_score,
                "quality_level": article_score.quality_level.value,
                "top_strengths": article_score.get_top_strengths(3),
                "top_weaknesses": article_score.get_top_weaknesses(3),
            }
        )
    return results
```

## Best Practices

### 1. Consistent Evaluation

- Use standardized criteria descriptions
- Apply consistent scoring across evaluators
- Document evaluation methodology

### 2. Weight Calibration

- Test different weight configurations
- Validate against known high/low quality examples
- Adjust weights based on domain requirements

### 3. Quality Monitoring

- Track score distributions over time
- Monitor for scoring drift or bias
- Regular calibration with human assessments

### 4. Integration Patterns

```python
# Integration with content pipeline
class ContentProcessor:
    def __init__(self):
        self.weights = WeightConfig()  # Use appropriate weights

    def process_article(self, article):
        # Extract or compute criteria scores
        scores = self.extract_criteria_scores(article)

        # Create assessment
        assessment = ArticleScore(
            scores=scores,
            weights=self.weights,
            assessment_summary=self.generate_summary(scores),
        )

        # Store results
        self.store_assessment(article.id, assessment.export_to_dict())

        return assessment
```

## Validation and Testing

The article scoring system includes comprehensive unit tests covering:

- **Model Validation**: Pydantic model constraints and validation
- **Score Calculation**: Mathematical accuracy of scoring algorithms
- **Edge Cases**: Boundary conditions and error handling
- **Integration**: End-to-end workflow testing

Run tests with:
```bash
pytest bertrend/tests/test_article_scoring_models.py -v
```

## Extension and Customization

### Adding New Criteria

1. **Update `CriteriaScores`** model with new fields
2. **Add corresponding weights** in `WeightConfig`
3. **Update validation logic** as needed
4. **Add comprehensive tests** for new functionality

### Custom Quality Levels

Extend or modify quality levels by updating the `QualityLevel` enum and corresponding logic in the `quality_level` property.

### Alternative Scoring Algorithms

Implement custom scoring by subclassing `ArticleScore` and overriding the `final_score` computed field.

## Troubleshooting

### Common Issues

1. **Validation Errors**: Check that all scores are between 0.0 and 1.0
2. **Weight Sum Errors**: Ensure total weights don't exceed number of criteria
3. **Missing Fields**: Verify all required criteria scores are provided

### Debugging Tools

```python
# Validate individual components
scores = CriteriaScores(...)  # Will raise ValidationError if invalid
weights = WeightConfig(...)  # Will validate weight constraints

# Check intermediate calculations
breakdown = article_score.get_detailed_breakdown()
for criterion, data in breakdown.items():
    print(f"{criterion}: {data}")
```

This comprehensive scoring system provides a robust foundation for automated content quality assessment while maintaining flexibility for domain-specific customization.