from typing import Any, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing_extensions import override


class AgentCallbackHandler(BaseCallbackHandler):

    @override
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running"""
        print(f"***Prompt to LLM:***\n{prompts[0]}")
        print("*********")


    @override
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running"""
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("*********")