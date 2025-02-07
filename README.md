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


## License

This project is [The MIT License (MIT)](https://mit-license.org) licensed, see the [LICENSE](LICENSE) file for details.