# GeoAgent

GeoAgent extends GeoChat from one-shot generation into a small iterative reasoning loop.

## Architecture

1. Observation
   - The agent receives an image and a query.
   - The current observation, action history, and tool outputs are stored in `GeoAgentMemory`.

2. Planning
   - A lightweight planner prompt asks the model to output a single JSON control object.
   - The planner chooses among `tool`, `answer`, and `stop`.

3. Tool Use
   - Tools are explicit Python objects with `name`, `description`, and `run(...)`.
   - The default prototype includes a crop tool for region zooming.
   - Tool outputs can return a new image, which becomes the next observation.

4. Memory
   - `GeoAgentMemory` tracks decisions, observations, and the full step trace.
   - The memory persists across iterations within a single task and is returned with the final result.

5. Finalization
   - If the planner does not terminate early, a final answer pass summarizes the accumulated trace.

## Minimal API

```python
from geochat.agent import GeoAgent, CropTool

agent = GeoAgent.from_pretrained(
    model_path="/path/to/checkpoint",
    model_base="/path/to/base",
    tools=[CropTool()],
)

result = agent.run(
    "What is the object cluster near the coast?",
    image=image,
    max_steps=3,
)

print(result.answer)
print(result.memory.observations)
```

## Pseudocode

```text
memory <- empty
current_image <- input image

for step in 1..N:
    planner_output <- LLM(query, memory, tool_catalog, current_image)
    action <- parse_json(planner_output)

    if action == answer or action == stop:
        return final answer

    if action == tool:
        tool_result <- tool.run(current_image, memory, arguments)
        memory.add(tool_result)
        current_image <- tool_result.image if present

if no final answer:
    return LLM(query, memory, concise_answer_mode)
```

## Design Trade-offs

- The planner is intentionally text-based and JSON-driven instead of adding a separate neural policy head.
- Crop and zoom are implemented as simple image transforms first; more advanced detectors can be plugged in later.
- The current prototype keeps memory inside the process, which is enough for a single task but not for long-lived sessions.

## Failure Modes

- The planner may emit malformed JSON or invent unavailable tools.
- Cropping can amplify bad localization if the first region is wrong.
- The loop can over-reason on ambiguous queries, so `max_steps` should stay small.
- Intermediate quality is harder to measure than final-answer accuracy, so trajectory metrics need explicit logging.
