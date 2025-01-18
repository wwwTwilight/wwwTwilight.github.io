---
date: '2025-01-15T11:42:17+08:00'
draft: false
title: 'golang入门'
categories: ["学习"]
tags: ["后端" , "golang"]
---

简单记一下golang的语法，看看能不能速成

# 第一个HelloWorld

```go
package main // 入口文件

import "fmt" // 导入标准包

func main() {
    fmt.Println("Hello World")
}
```

运行：`go run main.go`或者`go run .`执行全部

顺带一提go的函数就是像上面那样子声明的

# 变量声明

```go
var a int = 10 // 声明变量a，类型为int，值为10，是最完整的形式
var b = 10 // 声明变量b，类型自动识别，这里就是int，值为10
c := 10 // 声明变量c，类型自动识别，这里就是int，值为10，这种形式只能在函数里面这样子写，在函数外面只能用var，也就是上一种形式
```

# 常量

```go
const a = 10
const (
    a = 10
    b = 20
)
```

# 数据类型

```go
// 整型
int
int8
int16
int32
int64
uint
uint8
// 直接用int就行

float32
float64

bool

string

rune // int32
byte // uint8

// 复数
complex64
complex128
```

# 输出

```go
fmt.Println("Hello World") // 输出并换行
fmt.Printf("Hello World\n") // 输出并换行
fmt.Printf("a 的类型是 %T\n", a) // 输出a的类型
fmt.Printf("a 的值是 %v\n", a) // 输出a的值
```

占位符内容详细参考[这里](https://www.cnblogs.com/zxingduo/p/17662263.html)

# 运算符

```go
// 算术运算符
+ - * / % ++ --

// 关系运算符
== != > < >= <=

// 逻辑运算符
&& || !

// 位运算符
& | ^ << >>
```

# 条件语句

```go
// if
func main() {
    if a > 10 {
        fmt.Println("a 大于 10")
    } else if a < 10 {
        fmt.Println("a 小于 10")
    } else {
        fmt.Println("a 等于 10")
    }
}

// 可以在if中初始化变量，方式如下，只支持在这个if下使用
if a := 10; a > 10 {
    fmt.Println("a 大于 10")
} else if a < 10 {
    fmt.Println("a 小于 10")
} else {
    fmt.Println("a 等于 10")
}

// switch case
func main() {
    // 比起C++，这个相当于自带break
    switch a {
    case 1:
        fmt.Println("a 等于 1")
    case 2:
        fmt.Println("a 等于 2")
    default:
        fmt.Println("a 不等于 1 和 2")
    }
}

```

# 循环语句
go只有for循环
```go
// for
func main() {
    for i := 0; i < 10; i++ {
        fmt.Println(i)
    }
}

// for range
func main() {
    // 这个相当于C++的for each
    for i, v := range arr {
        fmt.Println(i, v)
    }
}
```

# 函数

```go
// 声明函数
func 函数名(a int, b int) 返回值列表 { // 返回值可以省略
    // 函数体
    return 返回值
}

// 返回多个值
func 函数名(a int, b int) (int, int) {
    // 函数体
    return b, a
}

// 命名返回值
func 函数名(参数列表) (返回值列表) {
    // 函数体
    return 返回值
}

// 可变参数
func 函数名(参数列表 ...参数类型) {
    
}

// 函数可以作为值放到变量中
func sum(a int, b int, transform func(int) int) int {
    return transform(a) + transform(b)
}
```

# 数组

```go
// 声明数组
var arr [5]int // 声明一个长度为5的int数组
arr := [5]int{1, 2, 3, 4, 5} // 声明并初始化一个长度为5的int数组，长度是不可变的
arr := [...]int{1, 2, 3, 4, 5} // 声明并初始化一个长度为5的int数组，长度由编译器自动计算
```

slice可以像vector一样
```go
a := make([]int, 5) // 声明一个长度为5的int数组
a := []int{1, 2, 3, 4, 5} // 声明并初始化一个长度为5的int数组
a = append(a, 1, 2, 3) // 在数组后面添加元素
a[0] = 5 // 修改数组元素
```

# map

```go
m := map[string]int{ //[key]value
    "a": 1,
    "b": 2,
    "c": 3,
}
m := make(map[string]int) // 声明一个map，没有初始值
```

# 结构体

```go
type Person struct {
    Name string
    Age int
}
// 方法
func (p Person) SayHello() { // 第一个参数相当于是this指针
    fmt.Println("Hello, my name is", p.Name, "and I am", p.Age, "years old.")
}
func (p *Person) SetName(name string) { // 这个函数不是定义在结构体内的，因此使用的时候如果涉及赋值操作，需要传递引用
    p.Name = name
}
// 初始化结构体
p := Person{
    Name: "Tom",
    Age: 18,
}
// 访问结构体成员
p.Name
p.Age
// 修改结构体成员
p.Name = "Jerry"
// 调用结构体方法
p.SayHello()
p.SetName("Jerry")
```

# 指针

```go
a := 10
b := &a // b是指向a的指针
*b = 20 // 修改指针指向的值
fmt.Println(a) // 输出20
// 或者
var a int= 20   /* 声明实际变量 */
var ip *int        /* 声明指针变量 */
ip = &a  /* 指针变量的存储地址 */
```

# 接口

```go
type Animal interface {
    SayHello() // 声明一个接口
}

type Dog struct {
    Name string
}

type Cat struct {
    Name string
}

func (d Dog) SayHello() {
    fmt.Println("Hello, my name is", d.Name)
}

func (c Cat) SayHello() {
    fmt.Println("Hello, my name is", c.Name)
}

func main() {
    var a Animal
    var b Animal
    a = Dog{Name: "Tom"}
    b = Cat{Name: "Jerry"}
    a.SayHello()
    b.SayHello()
}
```

# 错误处理
```go
func main() {
    n, err := fmt.Println("dd")
    if err != nil {
        // 正常
    } else {
        // 异常
    }
}
```

# 并发
```go
func main() {
    go func2()
    func1()
}

func func1() {
    time.Sleep(500 * time.Second)
    fmt.Println("func1")
}

func func2() {
    fmt.Println("func2")
}
```
channels管道通信，这个ch有点像是栈，后进先出
```go
func func1(ch chan string){
    ch <- "func1"
}
func func2(ch chan string){
    ch <- "func2"
}
func main() {
    ch := make(chan string)
    go func1(ch)
    res1 := <-ch
    go func2(ch)
    res2 := <-ch
    fmt.Println(res1)
    fmt.Println(res2)
}
```
结果是：func1 func2

```go
func func1(ch chan string) {
	ch <- "func1"
}
func func2(ch chan string) {
	ch <- "func2"
}
func main() {
	ch := make(chan string)
	go func1(ch)
	go func2(ch)
	res1 := <-ch
	res2 := <-ch
	fmt.Println(res1)
	fmt.Println(res2)
}
```
结果是：func2 func1