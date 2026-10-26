# Ragas compatibility patch: stubs deprecated VertexAI import in langchain_community
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
