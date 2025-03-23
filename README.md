# toy-llm
A LLM repo for learning

文档地址：[https://henryzhuhr.github.io/toyllm/](https://henryzhuhr.github.io/toyllm/)


## 开发容器使用

根据当前目录下的 `docker-compose.yml` 文件启动容器
```shell
docker compose up -d
docker compose up -d --build            # 重新构建镜像
docker compose up -d --force-recreate   # 强制重新创建容器
docker compose up -d --build --force-recreate
```

如果希望进入容器内部，可以使用以下命令
```shell
docker compose exec toyllm-development-env /bin/bash # 进入开发环境容器
```


停止并删除所有与当前 `docker-compose.yml` 文件关联的容器、网络和卷
```shell
docker compose down
```

组合上述开发命令

```shell
docker compose up -d --build --force-recreate && \
docker compose exec toyllm-development-env /bin/bash && \
docker compose down
```

- [microsoft/TaskWeaver](https://github.com/microsoft/TaskWeaver)


# ReAct Agent

- paper: [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629)
- paper prompt: [ReAct Prompting](https://github.com/ysymyth/ReAct)
- code: [langchain-ai/react-agent](https://github.com/langchain-ai/react-agent)
- Tutorial:
  - [手把手教你从零搭建Agent框架](https://github.com/OceanPresentChao/llm-ReAct/blob/main/doc/手把手教你从零搭建Agent框架.md)
  - [KMnO4-zx/TinyAgent](https://github.com/KMnO4-zx/TinyAgent): 基于ReAct手搓一个Agent Demo


## License

This project is [The MIT License (MIT)](https://mit-license.org) licensed, see the [LICENSE](LICENSE) file for details.