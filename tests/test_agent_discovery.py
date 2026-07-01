"""Guards the contract `adk web` / `adk run` rely on: a discoverable root_agent."""
from google.adk.agents import SequentialAgent


def test_root_agent_is_exposed_for_adk():
    import crucible
    assert hasattr(crucible.agent, "root_agent")
    root = crucible.agent.root_agent
    assert isinstance(root, SequentialAgent)
    assert root.name == "crucible"
    assert len(root.sub_agents) == 3


def test_judge_has_structured_output_schema():
    from crucible.rubric import JudgeVerdict
    root = __import__("crucible").agent.root_agent
    judge = root.sub_agents[-1]
    assert judge.name == "judge"
    assert judge.output_schema is JudgeVerdict
