from typing import Optional, List, Dict, Any
from src.model import ModelInterface
from src.tools import ToolRegistry, ToolResult
from src.parser import ModelOutputParser, Action
from src.trajectory import Trajectory, TrajectoryStep


class AgentHarness:
    """
    Minimal, model-agnostic execution harness.
    
    Acts strictly as a thin execution layer:
    - Maintains message history and conversation trajectory.
    - Sends available tool schemas to the model.
    - Parses model outputs into actions.
    - Executes requested tool calls via ToolRegistry.
    - Returns observations / error feedback to the model.
    - Enforces safety limits (max_steps).
    
    IMPORTANT:
    The harness NEVER chooses tools, infers intent, rewrites parameters,
    or guides decision-making. The model determines all actions autonomously.
    """

    def __init__(
        self,
        model: ModelInterface,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
        parser: Optional[ModelOutputParser] = None,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.parser = parser or ModelOutputParser()

    def run(self, prompt: str, task_id: str = "task_001") -> Trajectory:
        """
        Execute the agent loop for a given prompt until a final answer is produced
        or max_steps is exceeded.
        """
        trajectory = Trajectory(task_id=task_id, prompt=prompt)
        
        # Initial message history
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        tool_schemas = self.tool_registry.schemas()

        for step_idx in range(self.max_steps):
            # 1. Query the model with history & tool schemas
            model_output = self.model.generate(messages=messages, tools=tool_schemas)

            # 2. Parse model output into structured action
            action: Action = self.parser.parse(model_output)

            # 3. Handle Final Answer
            if action.is_final_answer():
                step = TrajectoryStep(
                    step=step_idx,
                    model_output=model_output,
                    action_type="final",
                )
                trajectory.add_step(step)
                trajectory.final_answer = action.content
                trajectory.termination_reason = "final_answer"

                # Append assistant message to trajectory history
                messages.append({"role": "assistant", "content": model_output})
                trajectory.messages = messages
                return trajectory

            # 4. Handle Parsing Error
            if action.is_parse_error():
                error_obs = f"PARSE_ERROR: {action.error_message}\nPlease respond with a valid JSON tool_call or final answer."
                step = TrajectoryStep(
                    step=step_idx,
                    model_output=model_output,
                    action_type="parse_error",
                    error=action.error_message,
                    tool_result=error_obs,
                )
                trajectory.add_step(step)

                # Feed error observation back to model to allow self-correction
                messages.append({"role": "assistant", "content": model_output})
                messages.append({"role": "user", "content": error_obs})
                continue

            # 5. Handle Tool Call
            if action.is_tool_call():
                tool_name = action.tool_name or ""
                arguments = action.arguments or {}

                # Execute requested tool via ToolRegistry (handles unknown tool & invalid arg checks)
                tool_result: ToolResult = self.tool_registry.execute(tool_name, arguments)
                observation_str = tool_result.to_observation_string()

                step = TrajectoryStep(
                    step=step_idx,
                    model_output=model_output,
                    action_type="tool_call",
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_result=observation_str,
                    error=tool_result.error if not tool_result.success else None,
                )
                trajectory.add_step(step)

                # Append assistant response & tool observation to message history
                messages.append({"role": "assistant", "content": model_output})
                messages.append({"role": "user", "content": observation_str})
                continue

        # 6. Max Steps Exceeded
        trajectory.termination_reason = "max_steps_exceeded"
        trajectory.messages = messages
        return trajectory
