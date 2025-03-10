---
date: '2025-03-10T07:39:53+08:00'
draft: false
title: 'SQL注入攻击'
categories: ["学习"]
tags: ["SQL注入攻击"]
---

# SQL 注入攻击及BERT模型

## SQL 注入攻击原理

SQL注入攻击（SQL Injection, SQLi）是一种发生在应用程序和数据库层的安全漏洞，是一种常用的数据库攻击手段，其利用特殊构造的SQL语句进行拼接，从而执行恶意SQL代码，如果在设计程序的时候没有进行防备，通过这些语句，黑客可以获取到数据库的数据以及其中的信息，甚至可以修改数据库中的数据，导致数据库遭到严重的破坏。

<center><img src = "../pics/SQLi示意图.jpg"/></center>

## 案例

常见的SQL注入攻击方式有：

1. 内联SQL注入
通过注释等方式，跳过某些验证，例如一个登陆的查询语句
```sql
SELECT * FROM users WHERE username = 'admin' AND password = '123'
```
如果用户输入的用户名是admin' AND 1=1 -- ，密码随意，那么拼接后的SQL语句就变成了
```sql
SELECT * FROM users WHERE username = 'admin' AND 1=1 -- ' AND password = '123'
```
因为`--` 是SQL中的注释符号，后面的内容都会被注释掉，所以密码这一项就不再起作用了，这样就可以绕过密码的验证，直接登陆到系统中。

2. 盲SQL注入

通过改变查询和观察页面内容变化，判断SQL查询是否成功执行，例如一个查询语句
```sql
SELECT * FROM products WHERE name LIKE '%[用户输入]%';
```
这时候如果攻击者想要知道是否存在一个名为users的表，可以通过改变查询语句，观察页面内容的变化来判断，通过输入以下内容
```sql
a' AND (SELECT 'test' FROM users) = 'test' –-
```
如果这之后页面返回正常内容，那么说明存在users表，否则说明不存在。

3. 联合查询SQL注入

通过UNION SELECT语句，攻击者可以进行额外的查询，将查询结果合并到一起，例如一个查询语句
```sql
SELECT * FROM products WHERE name = '用户输入';
```

此时，攻击者可以输入以下内容（假设数据类型能够匹配）
```sql
' UNION SELECT username, password FROM users –-
```
这样就可以将users表中的内容也显示出来。

## 当前主流的防御方法

1. 预处理语句：通过参数化查询，将SQL代码和用户输入严格分离，防止SQL注入攻击。
```go
stmt, err := db.Prepare("SELECT * FROM users WHERE username = ? AND password = ?")
if err != nil {
    log.Fatal(err)
}
rows, err := stmt.Query(username, password)
```

2. 使用对象关系映射框架：通过ORM框架，将SQL语句和用户输入严格分离，防止SQL注入攻击。
```go
var user User
db.Where("username = ? AND password = ?", username, password).First(&user)
```

3. 输入验证：对用户输入进行严格的限制和验证（比如禁止输入'' -- #等字符），防止SQL注入攻击。

4. 最小权限原则：通过限制数据库用户的权限，使得即使发生了SQL注入攻击，攻击者也无法对数据库进行过度的操作。

但是上述的所有方法都有无法防御的情况，因此在实际的使用中，需要结合多种方法进行防御。

# BERT模型与SQL注入攻击

## BERT模型简介

BERT（Bidirectional Encoder Representations from Transformers）是一种基于Transformer的预训练语言模型，其目的在于完成自然语言理解任务。它通过双向编码器来学习语言中的上下文信息，从而实现双向上下文理解，能够同时考虑句子中的前后文信息，从而更好地理解语言语义。

## BERT模型在SQL注入攻击中的应用

BERT模型应对SQL注入攻击的主要思想是利用BERT模型的自然语言处理能力，攻击者通过恶意的SQL代码破坏数据库，而BERT模型可以通过将攻击检测转换为NLP任务，让BERT识别输入是否包含SQL注入攻击。

具体而言，首先需要收集SQL查询数据，其中包含正常的SQL语句和恶意的SQL注入语句，将其进行分词和向量化，转换为BERT模型可以处理的输入形式，然后通过BERT模型进行训练，使其能够识别出恶意的SQL注入语句。

<center><img src = "../pics/BERT训练.png"/></center>
训练完成后，将待检测的SQL语句输入到BERT模型中，模型会输出一个概率值，表示该SQL语句是否包含SQL注入攻击，从而实现对SQL注入攻击的检测。

## BERT模型在SQL注入攻击检测中的优势

相比传统的应对SQL注入攻击的方法，BERT模型具有明显的优势
1. 由于BERT具有强大的自然语言处理能力，能够更好地理解SQL语句的语义，相比传统的防御方式，能够更准确地识别出SQL注入攻击以及各种攻击的变种。
2. 某些SQLi的攻击方式复杂，例如使用/**/注释符，传统的防御方式难以识别，而BERT模型可以更好地处理这些复杂的SQL注入攻击。
3. 传统方法在某些规则方面较为严格，可能会误判常规的SQL语句，而BERT由于可以理解SQL语句的语义，可以更好地识别出SQL注入攻击，降低误判率。

## BERT模型在SQL注入攻击检测中的局限性

尽管BERT在SQL注入攻击检测中表现出显著的优势，但是也存在一些局限性
1. BERT需要大量的训练数据，而收集高质量的SQL注入攻击数据是一项非常困难的工作。
2. BERT模型需要大量的计算资源，对于实时性要求较高的场景，BERT模型可能无法满足需求。
3. BERT模型只提供一个概率值，并不能直接判断SQL语句是否包含SQL注入攻击，需要根据概率值进行进一步的判断和处理。

<table>
  <thead>
    <tr>
      <th>防御方法</th>
      <th>误报率</th>
      <th>维护成本</th>
      <th>对变种 SQLi 的防御能力</th>
      <th>执行效率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>参数化查询</td>
      <td>低</td>
      <td>低</td>
      <td>弱</td>
      <td>高</td>
    </tr>
    <tr>
      <td>ORM</td>
      <td>低</td>
      <td>低</td>
      <td>弱</td>
      <td>中等</td>
    </tr>
    <tr>
      <td>输入验证</td>
      <td>中</td>
      <td>低</td>
      <td>弱</td>
      <td>高</td>
    </tr>
    <tr>
      <td>BERT</td>
      <td>低</td>
      <td>高</td>
      <td>强</td>
      <td>低</td>
    </tr>
  </tbody>
</table>
