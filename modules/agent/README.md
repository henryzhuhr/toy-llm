- [microsoft/TaskWeaver](https://github.com/microsoft/TaskWeaver)


# ReAct Agent

- paper: [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629)
- paper prompt: [ReAct Prompting](https://github.com/ysymyth/ReAct)
- code: [langchain-ai/react-agent](https://github.com/langchain-ai/react-agent)
- Tutorial:
  - [手把手教你从零搭建Agent框架](https://github.com/OceanPresentChao/llm-ReAct/blob/main/doc/手把手教你从零搭建Agent框架.md)
  - [KMnO4-zx/TinyAgent](https://github.com/KMnO4-zx/TinyAgent): 基于ReAct手搓一个Agent Demo


%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
        __start__([<p>__start__</p>]):::first
        planner(planner)
        executor(executor)
        replanner(replanner)
        __end__([<p>__end__</p>]):::last
        __start__ --> planner;
        executor --> replanner;
        planner --> executor;
        replanner -.-> executor;
        replanner -.-> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc