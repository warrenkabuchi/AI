import pytest

from copilot.question import looks_like_question


@pytest.mark.parametrize("text", [
    "What is Kubernetes?",
    "What is Kubernetes",           # whisper often drops the question mark
    "how would you scale this service",
    "Can you walk me through your CI pipeline",
    "Explain the difference between ECS and EKS.",
    "Really?",
])
def test_questions(text):
    assert looks_like_question(text)


@pytest.mark.parametrize("text", [
    "",
    "I deployed it yesterday.",
    "We use Terraform for everything.",
    "The what-if analysis is done.",  # interrogative not in first position
])
def test_non_questions(text):
    assert not looks_like_question(text)
