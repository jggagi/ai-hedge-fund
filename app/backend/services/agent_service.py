from functools import partial
from typing import Callable
from src.graph.state import AgentState
from src.run_context import RunExecutionContext

def create_agent_function(
    agent_function: Callable,
    agent_id: str,
    run_context: RunExecutionContext | None = None,
) -> Callable[[AgentState], dict]:
    """
    Creates a new function from an agent function that accepts an agent_id.

    :param agent_function: The agent function to wrap.
    :param agent_id: The ID to be passed to the agent.
    :return: A new function that can be called by LangGraph.
    """
    bound_agent = partial(agent_function, agent_id=agent_id)

    def guarded_agent(state: AgentState) -> dict:
        context = run_context or state.get("metadata", {}).get("run_context")
        if context is not None:
            context.check_active()
        return bound_agent(state)

    return guarded_agent
