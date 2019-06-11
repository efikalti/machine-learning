# =============================================================================
# HOMEWORK 2 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# Import Orange library
import Orange


# Define evaluators
entropy = Orange.classification.rules.EntropyEvaluator
laplace = Orange.classification.rules.LaplaceAccuracyEvaluator
wracc = Orange.classification.rules.WeightedRelativeAccuracyEvaluator

# Model variables
evaluators = [[entropy, "Entropy"], [laplace, "Laplace"], [wracc, "WRACC"]]
learner_type = ["Ordered", "Unordered"]
beams = [5, 10]
min_covered = [7, 15]
max_rule_lengths = [2, 5]
average = "macro"

# Load 'wine' dataset
wineData = Orange.data.Table("wine")


for type in learner_type:
    # Iterate all learner types
    for evaluator in evaluators:
        # Iterate all evaluators
        for beam in beams:
            # Iterate all beams
            for min in min_covered:
                # Iterate all values for min covered min covered examples
                for length in max_rule_lengths:
                    # Iterate all max rule lengths

                    # Create rule model for these variables
                    learner = None
                    if type == "Ordered":
                        learner = Orange.classification.rules.CN2Learner()
                    else:
                        learner = Orange.classification.rules.CN2UnorderedLearner()
                    learner.rule_finder.search_algorithm.evaluator = evaluator[0]
                    learner.rule_finder.search_algorithm.beam_width = beam
                    learner.rule_finder.search_algorithm.min_covered_examples = min
                    learner.rule_finder.general_validator.max_rule_length = length

                    # Test the trained model
                    results = Orange.evaluation.CrossValidation(wineData, [learner])

                    # Print precision, recall and f1 metrics for the results with the defined average
                    print('# =============================================================================')
                    print('Learner type:                     ' + type)
                    print('Evaluator:                        ' + evaluator[1])
                    print('Beam width:                       ' + str(beam))
                    print('Minimum examples covered:         ' + str(min))
                    print('Maximum rule length:              ' + str(length))
                    print('Average:                          ' + str(average))
                    print('Results:')
                    print("Precision:               %.3f" % Orange.evaluation.Precision(results, average=average)[0])
                    print("Recall:                  %.3f" % Orange.evaluation.Recall(results, average=average)[0])
                    print("F1:                      %.3f" % Orange.evaluation.F1(results, average=average)[0])
                    print()
                    print('Rules:')

                    # Train learner manually
                    classifier = learner(wineData)


                    # Print the derived rules
                    for rule in classifier.rule_list:
                        print(rule)

                    print()
