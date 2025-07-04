---
date: '2025-06-29T22:07:09+08:00'
draft: false
title: 'C++WebServer开发日志（已完结）'
---

先从读现有的代码开始，这是参考大佬的仓库
[基于c语言的web服务器](https://github.com/forthespada/MyPoorWebServer)

# 额外知识

## http请求报文包含的内容

### 请求行

请求行由三部分组成：请求方法、请求URL和HTTP协议版本。

例如：

![](http://www.itheima.com/images/newslistPIC/1692776571417_%E8%AF%B7%E6%B1%82%E8%A1%8C.png)

### 请求头

请求头由一系列的键值对组成，每个键值对之间用冒号分隔。请求头用于向服务器传递额外的信息，例如客户端的浏览器类型、语言、编码方式等。
![](http://www.itheima.com/images/newslistPIC/1692777388813_%E5%B8%B8%E8%A7%81%E7%9A%84%E8%AF%B7%E6%B1%82%E5%A4%B4%E5%AD%97%E6%AE%B5.png)

### 空行

请求头和请求体之间必须有一个空行，用于分隔请求头和请求体。

### 请求体

请求体是可选的，用于向服务器传递额外的数据。例如，当客户端向服务器发送POST请求时，请求体中通常会包含表单数据或JSON数据。

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

## cgi文件是

CGI（Common Gateway Interface，通用网关接口）是一种用于在Web服务器上执行外部程序的标准接口。CGI程序通常用于处理用户提交的表单数据，生成动态内容，并与数据库进行交互等。

当内容是动态的时候，服务器会调用cgi文件，cgi文件会生成动态内容，然后返回给服务器，服务器再返回给客户端

### cgi中的环境变量

CGI协议规定了一些**标准环境变量**，CGI脚本通过这些变量获取HTTP请求信息。常见的有：

| 变量名            | 说明                                   |
|-------------------|----------------------------------------|
| REQUEST_METHOD    | 请求方法（GET、POST等）                |
| QUERY_STRING      | URL中的查询字符串（GET参数）           |
| CONTENT_LENGTH    | 请求体长度（POST时有用）               |
| CONTENT_TYPE      | 请求体类型（如 application/x-www-form-urlencoded）|
| SCRIPT_NAME       | 脚本路径                               |
| SERVER_NAME       | 服务器主机名                           |
| SERVER_PORT       | 服务器端口                             |
| SERVER_PROTOCOL   | 协议版本（如 HTTP/1.1）                |
| REMOTE_ADDR       | 客户端IP地址                           |
| HTTP_*            | 以HTTP_开头的变量，映射所有HTTP请求头  |

这些变量不是随便命名的，而是CGI协议规定的，Web服务器要负责设置好，CGI脚本才能正确获取。

## stat结构体的使用

### stat结构体

stat结构体是C语言中用于描述文件状态的结构体，它包含了文件的权限、大小、修改时间等信息。stat结构体定义在头文件`<sys/stat.h>`中。

### stat函数

stat函数用于获取文件的状态信息，它的原型如下：

```cpp
#include <sys/stat.h>

int stat(const char *path, struct stat *buf);
```

参数说明：
- path：要获取状态的文件路径。
- buf：指向stat结构体的指针，用于存储文件的状态信息。

返回值：
- 成功时，返回0。
- 失败时，返回-1，并设置errno变量。

### 使用示例

```cpp
#include <iostream>
#include <sys/stat.h>

int main() {
    struct stat fileStat;
    if (stat("example.txt", &fileStat) == 0) {
        std::cout << "File size: " << fileStat.st_size << " bytes" << std::endl;
        std::cout << "Last modified: " << fileStat.st_mtime << std::endl;
    } else {
        std::cerr << "Failed to get file status" << std::endl;
    }
    return 0;
}
```

除了确认文件是否存在，还可以确定文件的类型，比如是普通文件、目录、符号链接等。这些信息都存放在stat结构体的st_mode字段中，需要读取这些信息需要使用相对应的掩码

### 常见的文件类型掩码及权限掩码（`st_mode`相关）

#### 1. 文件类型掩码（判断文件类型）

| 掩码/类型常量   | 含义        | 判断方式示例                                      |
|-----------------|-------------|---------------------------------------------------|
| `S_IFMT`        | 类型掩码    | `st.st_mode & S_IFMT`                             |
| `S_IFREG`       | 普通文件    | `if ((st.st_mode & S_IFMT) == S_IFREG)`           |
| `S_IFDIR`       | 目录        | `if ((st.st_mode & S_IFMT) == S_IFDIR)`           |
| `S_IFLNK`       | 符号链接    | `if ((st.st_mode & S_IFMT) == S_IFLNK)`           |
| `S_IFCHR`       | 字符设备    | `if ((st.st_mode & S_IFMT) == S_IFCHR)`           |
| `S_IFBLK`       | 块设备      | `if ((st.st_mode & S_IFMT) == S_IFBLK)`           |
| `S_IFIFO`       | 管道/FIFO   | `if ((st.st_mode & S_IFMT) == S_IFIFO)`           |
| `S_IFSOCK`      | 套接字      | `if ((st.st_mode & S_IFMT) == S_IFSOCK)`          |

###### 示例代码
```c
#include <sys/stat.h>
struct stat st;
stat("test.txt", &st);

if ((st.st_mode & S_IFMT) == S_IFREG) {
    // 普通文件
}
if ((st.st_mode & S_IFMT) == S_IFDIR) {
    // 目录
}
if ((st.st_mode & S_IFMT) == S_IFLNK) {
    // 符号链接
}
```

#### 2. 文件权限掩码（判断权限）

| 掩码/常量      | 含义                 | 判断方式示例                        |
|----------------|----------------------|-------------------------------------|
| S_IRUSR      | 所有者可读           | if (st.st_mode & S_IRUSR)         |
| S_IWUSR      | 所有者可写           | if (st.st_mode & S_IWUSR)         |
| S_IXUSR      | 所有者可执行         | if (st.st_mode & S_IXUSR)         |
| S_IRGRP      | 用户组可读           | if (st.st_mode & S_IRGRP)         |
| S_IWGRP      | 用户组可写           | if (st.st_mode & S_IWGRP)         |
| S_IXGRP      | 用户组可执行         | if (st.st_mode & S_IXGRP)         |
| S_IROTH      | 其他用户可读         | if (st.st_mode & S_IROTH)         |
| S_IWOTH      | 其他用户可写         | if (st.st_mode & S_IWOTH)         |
| S_IXOTH      | 其他用户可执行       | if (st.st_mode & S_IXOTH)         |

###### 示例代码
```c
if (st.st_mode & S_IRUSR) {
    // 文件所有者有读权限
}
if (st.st_mode & S_IXUSR) {
    // 文件所有者有执行权限
}
if (st.st_mode & S_IXGRP) {
    // 文件所属组有执行权限
}
if (st.st_mode & S_IXOTH) {
    // 其他用户有执行权限
}
```

### 3. 综合实用案例

#### 判断文件类型和权限
```c
#include <sys/stat.h>
#include <stdio.h>

struct stat st;
if (stat("test.txt", &st) == 0) {
    if ((st.st_mode & S_IFMT) == S_IFREG) {
        printf("普通文件\n");
    }
    if ((st.st_mode & S_IFMT) == S_IFDIR) {
        printf("目录\n");
    }
    if ((st.st_mode & S_IFMT) == S_IFLNK) {
        printf("符号链接\n");
    }
    if (st.st_mode & S_IXUSR) {
        printf("所有者可执行\n");
    }
    if (st.st_mode & S_IRUSR) {
        printf("所有者可读\n");
    }
    if (st.st_mode & S_IWUSR) {
        printf("所有者可写\n");
    }
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

### 端口复用的概念

为什么需要端口复用？

在服务器端，一个端口只能被一个进程监听，当一个进程关闭后，这个端口才能被其他进程监听。但是，如果这个进程没有正常关闭，而是被强制终止，那么这个端口就会一直被占用，导致其他进程无法监听这个端口。使用端口复用，可以避免这种情况。通过设置 SO_REUSEADDR 套接字选项，可以让一个进程在关闭后立即释放端口，从而让其他进程能够立即使用这个端口。

如何设置端口复用？

主要用到了这个函数`setsockopt()`，设置 SO_REUSEADDR 套接字选项，可以让一个进程在关闭后立即释放端口，从而让其他进程能够立即使用这个端口。

```cpp
int setsockopt(int sockfd, int level, int optname, const void *optval, socklen_t optlen);
```
参数说明：
- sockfd：需要设置选项的套接字文件描述符。
- level：选项所在的协议层。对于 SO_REUSEADDR 选项，level 应设置为 SOL_SOCKET。
- optname：需要设置的选项名称。对于 SO_REUSEADDR 选项，optname 应设置为 SO_REUSEADDR。
- optval：指向包含选项值的缓冲区的指针。对于 SO_REUSEADDR 选项，optval 应设置为非零值。
- optlen：optval 缓冲区的大小。

返回值：
- 成功时，返回 0。

## 网络字节序和主机字节序转换

在网络编程中，不同主机之间的字节序可能不同，因此需要进行字节序转换。网络字节序是大端字节序，而主机字节序可能是小端字节序或大端字节序。因此，在进行网络编程时，需要将主机字节序转换为网络字节序，或者将网络字节序转换为主机字节序。

以下是常用的转换，由copilot生成

### 1. 主机字节序 ↔️ 网络字节序

| 函数名      | 功能说明                                   | 作用对象         |
|-------------|--------------------------------------------|------------------|
| htons     | Host TO Network Short                      | 16位（2字节）无符号整数（如端口号）|
| htonl     | Host TO Network Long                       | 32位（4字节）无符号整数（如IPv4地址）|
| ntohs     | Network TO Host Short                      | 16位（2字节）无符号整数（如端口号）|
| ntohl     | Network TO Host Long                       | 32位（4字节）无符号整数（如IPv4地址）|

#### 具体说明
- `htons(x)`：主机字节序的16位整数转为网络字节序（常用于端口号）。
- `htonl(x)`：主机字节序的32位整数转为网络字节序（常用于IPv4地址）。
- `ntohs(x)`：网络字节序的16位整数转为主机字节序（常用于端口号）。
- `ntohl(x)`：网络字节序的32位整数转为主机字节序（常用于IPv4地址）。

---

### 2. IP地址字符串 ↔️ 二进制

| 函数名         | 功能说明                                   | 作用对象         |
|----------------|--------------------------------------------|------------------|
| inet_addr    | 点分十进制字符串转为网络字节序的IPv4地址   | "127.0.0.1" → uint32_t |
| inet_ntoa    | 网络字节序的IPv4地址转为点分十进制字符串   | uint32_t → "127.0.0.1" |
| inet_pton    | 字符串IP转为网络字节序（支持IPv4/IPv6）    | 推荐新项目使用    |
| inet_ntop    | 网络字节序IP转为字符串（支持IPv4/IPv6）    | 推荐新项目使用    |

#### 具体说明
- `inet_addr("127.0.0.1")`：字符串转为网络字节序的IPv4地址。
- `inet_ntoa(in_addr)`：网络字节序的IPv4地址转为字符串。
- `inet_pton(AF_INET, "127.0.0.1", &addr)`：更通用，支持IPv4/IPv6。
- `inet_ntop(AF_INET, &addr, buf, buflen)`：更通用，支持IPv4/IPv6。

---

### 总结

- **htons/htonl/ntohs/ntohl**：主机字节序和网络字节序之间的转换（端口号、IP地址等）。
- **inet_addr/inet_ntoa/inet_pton/inet_ntop**：IP地址的字符串和二进制之间的转换。

# 代码解读

注释版会写在[这里](https://github.com/wwwTwilight/myCppWebServer.git)，这篇文章写补充的内容