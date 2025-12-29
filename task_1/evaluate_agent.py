"""DeepEval Metrics Evaluation for E-Commerce Agent.

This module demonstrates how to evaluate the e-commerce agent using DeepEval metrics
for reasoning, action, and execution layers.

Authors:
    Anthony Edbert Feriyanto

References:
    NONE
"""

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from dotenv import load_dotenv

from task_1.agent import ecommerce_agent
from task_1.llm import call_llm
from task_2.gemini_evaluator import GeminiEvaluator

gemini_evaluator = GeminiEvaluator(model_name="gemini-2.5-flash")

load_dotenv()


def create_evaluation_dataset() -> EvaluationDataset:
    """Create evaluation dataset with golden test cases.

    Returns:
        EvaluationDataset: Dataset containing golden test cases for evaluation.
    """
    goldens = [
        Golden(
            input=(
                "I want to order 2 laptops to Jakarta, use discount code WELCOME10, "
                "pay with credit card, email: john.doe@example.com"
            ),
            expected_output=(
                "Order successfully placed. Transaction confirmed with 10% discount applied. "
                "Total amount charged to credit card. Confirmation email sent to john.doe@example.com."
            ),
        ),
        # Golden(
        #     input=(
        #         "Order 1 monitor and ship to Singapore, I have VIP20 discount code, "
        #         "payment via ewallet, send confirmation to vip.customer@company.com"
        #     ),
        #     expected_output=(
        #         "Order successfully placed. Transaction confirmed with 20% VIP discount applied. "
        #         "Total amount charged via e-wallet. Confirmation email sent to vip.customer@company.com."
        #     ),
        # ),
        # Golden(
        #     input=(
        #         "I need 5 headsets delivered to Surabaya, apply SAVE50 discount, "
        #         "pay by bank transfer, email: procurement@company.id"
        #     ),
        #     expected_output=(
        #         "Order successfully placed. Transaction confirmed with $50 discount applied. "
        #         "Total amount to be paid via bank transfer. Confirmation email sent to procurement@company.id."
        #     ),
        # ),
    ]

    return EvaluationDataset(goldens=goldens)


def evaluate_reasoning_layer():
    """Evaluate the reasoning layer of the agent.

    Uses PlanQualityMetric and PlanAdherenceMetric to assess the agent's
    planning and adherence to the plan.
    """

    plan_quality = PlanQualityMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)
    plan_adherence = PlanAdherenceMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)

    dataset = create_evaluation_dataset()

    print(f"\nEvaluating {len(dataset.goldens)} test cases...")
    print("\nMetrics:")
    print("  - PlanQualityMetric: Evaluates if the plan is logical, complete, and efficient")
    print("  - PlanAdherenceMetric: Evaluates if the agent follows its own plan")

    for golden in dataset.evals_iterator(metrics=[plan_quality, plan_adherence]):
        print(f"\n{'=' * 80}")
        print(f"Test Case: {golden.input[:80]}...")
        print(f"{'=' * 80}")

        result = ecommerce_agent(golden.input)

        print(f"\nAgent Result: {'Success' if result['success'] else 'Failed'}")


def evaluate_action_layer():
    """Evaluate the action layer (tool calling) of the agent.

    Uses ToolCorrectnessMetric and ArgumentCorrectnessMetric to assess
    tool selection and argument generation.
    """

    tool_correctness = ToolCorrectnessMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)
    argument_correctness = ArgumentCorrectnessMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)

    print("\nMetrics:")
    print("  - ToolCorrectnessMetric: Evaluates if correct tools are selected")
    print("  - ArgumentCorrectnessMetric: Evaluates if tool arguments are correct")


def evaluate_execution_layer():
    """Evaluate the overall execution of the agent.

    Uses TaskCompletionMetric and StepEfficiencyMetric to assess
    task completion and execution efficiency.
    """

    task_completion = TaskCompletionMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)
    step_efficiency = StepEfficiencyMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)

    dataset = create_evaluation_dataset()

    print(f"\nEvaluating {len(dataset.goldens)} test cases...")
    print("\nMetrics:")
    print("  - TaskCompletionMetric: Evaluates if the agent completes the task")
    print("  - StepEfficiencyMetric: Evaluates if the agent is efficient (no redundant steps)")

    for golden in dataset.evals_iterator(metrics=[task_completion, step_efficiency]):
        print(f"\n{'=' * 80}")
        print(f"Test Case: {golden.input[:80]}...")
        print(f"{'=' * 80}")

        result = ecommerce_agent(golden.input)

        print(f"\nAgent Result: {'Success' if result['success'] else 'Failed'}")



def evaluate_end_to_end():
    """Run end-to-end evaluation with all metrics.

    Combines reasoning, action, and execution metrics for comprehensive evaluation.
    """
    plan_quality = PlanQualityMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)
    plan_adherence = PlanAdherenceMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)
    task_completion = TaskCompletionMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)
    step_efficiency = StepEfficiencyMetric(threshold=0.7, include_reason=True, model=gemini_evaluator)

    dataset = create_evaluation_dataset()

    print(f"\nEvaluating {len(dataset.goldens)} test cases with all metrics...")
    print("\nMetrics:")
    print("  Reasoning Layer:")
    print("    - PlanQualityMetric")
    print("    - PlanAdherenceMetric")
    print("  Execution Layer:")
    print("    - TaskCompletionMetric")
    print("    - StepEfficiencyMetric")

    for golden in dataset.evals_iterator(
        metrics=[plan_quality, plan_adherence, task_completion, step_efficiency]
    ):
        print(f"\n{'=' * 80}")
        print(f"Test Case: {golden.input[:80]}...")
        print(f"{'=' * 80}")

        result = ecommerce_agent(golden.input)

        print(f"\nAgent Result: {'Success' if result['success'] else 'Failed'}")
        if result["success"]:
            payment = result.get("tools", {}).get("process_payment", {})
            print(f"Transaction ID: {payment.get('transaction_id', 'N/A')}")
            



if __name__ == "__main__":
    evaluate_end_to_end()
