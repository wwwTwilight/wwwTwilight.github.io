---
date: '2025-06-29T22:07:09+08:00'
draft: true
title: 'C++WebServer开发日志'
---

先从读现有的代码开始，这是参考大佬的仓库
[基于c语言的web服务器](https://github.com/forthespada/MyPoorWebServer)

# 额外知识

## C++中的线程是如何使用的

### 线程的创建与使用

```cpp
#include <iostream>
#include <thread>

void printMessage(int count) {
    for (int i = 0; i < count; ++i) {
        std::cout << "Hello from thread (function pointer)!\n";
    }
}

int main() {
    std::thread t1(printMessage, 5); // 创建线程，传递函数指针和参数
    t1.join(); // 等待线程完成，阻塞的
    t1.detach(); // 分离线程，不阻塞，主线程不会等待它执行完，可能主线程结束时子线程还在跑，甚至被杀死
    return 0;
}
```

```cpp
#include <iostream>
#include <thread>
#include <chrono>

void worker() {
    std::cout << "Worker thread started." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2)); // 模拟耗时操作
    std::cout << "Worker thread finished." << std::endl;
}

int main() {
    std::thread t(worker);

    std::cout << "Main thread is waiting for worker thread to finish..." << std::endl;
    t.join();  // 这里阻塞主线程
    std::cout << "Main thread resumes after join()." << std::endl;

    return 0;
}
```

## 套接字socket使用

### 原理说明

套接字可以看作一根管子，一端连接服务器，一端连接客户端，数据通过套接字在服务器和客户端之间传输。

服务端负责：
- 创建 socket
- 绑定端口和地址（bind）
- 等待连接（listen）
- 接收连接（accept）
- 收发数据（read/write）

客户端负责：
- 创建 socket
- 连接服务器（connect）
- 收发数据（send/recv）

```
流程图
客户端                            服务器
--------                        --------------
socket()                        socket()
                                bind()
                                listen()
connect()  ------------------>  accept()
send()/recv() <------------->  recv()/send()
close()                         close()
```

1. 应用程序调用 socket() 获取文件描述符
2. 操作系统在内核中创建 socket 对象
3. （服务器）bind() 把 socket 绑定地址端口
4. listen() 等待连接
5. （客户端）connect() 向服务器发起连接
6. accept() 接收连接，得到新 socket
7. 双方用 read()/write() 进行数据通信
8. 通信完成后 close() 关闭连接


# 代码解读

注释版会写在[这里](https://github.com/wwwTwilight/myCppWebServer.git)，这篇文章写补充的内容